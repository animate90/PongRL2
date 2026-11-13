import numpy as np


import gymnasium as gym


from pathlib import Path

import pickle

import torch

import Utilities
import BufferVectorized
import config
import Networks
import AtariEnv
import agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Create our training environmentS
envs = gym.vector.SyncVectorEnv(
    [AtariEnv.make_env(i) for i in range(config.numberEnvs)]
)
action_space_size = envs.action_space.nvec[0]

# 2. initilize Buffers
frameBuffers = BufferVectorized.FrameBuffer(config.numberEnvs , config.FrameBufferSize)

replayBuffer = BufferVectorized.ReplayBuffer(
    capacity=config.replayBufferCapacity,
    frameBufferSize=config.FrameBufferSize,
    batch_size=config.batch_size
)

# 3. initialize Deep Network
Network_Prediction = Networks.DQN(config.FrameBufferSize, action_space_size).to(device)

# load old weights
#path = Path(config.saveDir) / "modelWeights" / "model.pth"
#Network_Prediction.load_state_dict(torch.load(path, weights_only=True))

Network_Target = Networks.DQN(config.FrameBufferSize, action_space_size).to(device)
Network_Target.load_state_dict(Network_Prediction.state_dict())

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(Network_Prediction.parameters(), lr=2e-4)

# init agents
agents = agent.agents(envs)

# Training
# -------------------------------------------------------------


states, infos = envs.reset(seed=[100 + i for i in range(config.numberEnvs)])


#scores = Utilities.scoreKeeper(config.numberEnvs)

Data = Utilities.CollectData()

# adjust for vectoritdev env
newRange = float( config.final_exploration_iteration / config.numberEnvs )


for f in range(int(newRange)):
    # save state
    if frameBuffers.filled():
        states = frameBuffers.return_stack()

    # Agent -- choose action
    epsilon = Utilities.lerp(config.initial_epsilon, config.final_epsilon, f/ newRange)
    actions = agents.select_actions(epsilon, Network_Prediction, device, frameBuffers)

    next_states, rewards, truncateds, terminateds, infos = envs.step(actions)
    dones = np.logical_or(terminateds, truncateds)

    # add next state to frame buffer
    frameBuffers.addFrames(next_states)

    # Add to replay buffer
    if infos["episode_frame_number"][0] > 50:
        for envID in range(config.numberEnvs):
            replayBuffer.add(
                state= states[envID],
                action=actions[envID],
                reward=rewards[envID],
                next_state= frameBuffers.return_stack()[envID],
                done=terminateds[envID]
            )

    # score vector
    Data.addScore(rewards)

    # train network
    if replayBuffer.filled():
        # gradient descent for each env
        for step in range(config.numberEnvs):
            Buffer = replayBuffer.sample()
            states = Buffer["states"]
            next_states = Buffer["next_states"]
            rewards = Buffer["rewards"]
            actions = Buffer["actions"]
            dones = Buffer["dones"]

            Network_Prediction.zero_grad()

            # Current Q-values
            predicted_q = Network_Prediction(states.float().div(255)).gather(1, actions.unsqueeze(1)).squeeze(1)

            # DQN Target
            with torch.no_grad():
                best_actions = Network_Prediction(next_states.float().div(255)).argmax(dim=1, keepdim=True)
                next_q = Network_Target(next_states.float().div(255)).gather(1, best_actions).squeeze(1)
                target_q = rewards + (config.discount_factor * next_q * (1.0 - dones.float()))

                # td error
                td_error = target_q - predicted_q
                Data.addTDError( float(td_error.mean() ))

            loss = criterion(predicted_q, target_q)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # save data
            Data.addLosses(loss.item())
            Data.addAverageQValues(predicted_q.mean().item())
            Data.addEpsilon(epsilon)
        # update target net
        if f % config.target_network_update_freq == 0:
            print("!! update target network !!")
            Network_Target.load_state_dict(Network_Prediction.state_dict())

        if f % 100000000 == 0:
            print(
                f"Step: {f} | Input Range: [{states.min().item():.3f}, {states.max().item():.3f}]"
            )
            print(f"Mean Q-Value: {predicted_q.mean().item():.4f}")
            print(
                f"Target Q-Value: {target_q.mean().item():.4f} | Loss: {loss.item():.6f}"
            )

    # save scores
    if f % 20 == 0 :
        Data.appendAverageScores()
    # print
    if f % 1000 == 0 and f > 1:
        print("current iteration: {0} with epsilon {1}".format( f , epsilon) )
        print("")
        Data.PrintAverageScores()
        Data.PrintAverageLosses(50)
        Data.PrintAverageQvalues(50)
        Data.PrintAverageTDerror(50)
        print("replay current size: {0}".format(replayBuffer.currentSize) )
        print("-------------------------------------")
    if f % 50000 == 0 and f > 1:
        # save files
        replayBuffer.save_buffer(config.saveDir)
        Networks.saveNetwork(
            network=Network_Prediction, saveDir=config.saveDir, iteration=f
        )
        path = Path(config.saveDir) / "dataRecords"
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "records", "wb") as fout:
            pickle.dump(Data, fout)

