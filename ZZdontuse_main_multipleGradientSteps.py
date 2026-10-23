import gymnasium as gym

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import Utilities
import BufferVectorized
import config
import Networks
import AtariEnv
import agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Create our training environmentS
envs = gym.vector.SyncVectorEnv([AtariEnv.make_env(i) for i in range(config.numberEnvs)])
action_space_size = envs.action_space.nvec[0]
states, infos = envs.reset(seed=[100 + i for i in range(config.numberEnvs)])

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

#criterion = torch.nn.MSELoss()
criterion = torch.nn.SmoothL1Loss()

optim = torch.optim.Adam(Network_Prediction.parameters(), lr=2e-4)

# init agents
agents = agent.agents(envs)
epsilon = agent.epsilonDecay(config.epsilon_initial, config.epsilon_final, config.epsilon_decay_Start,config.epsilon_decay_End)

# Training
# -------------------------------------------------------------

# metrics
Scores = Utilities.Scores(numberEnvs=config.numberEnvs)
writer = SummaryWriter()


hparams = Utilities.get_clean_hparams(config)
result = "\n".join(f"{k}: {v}" for k, v in hparams.items())
writer.add_text("Hyperparameter", result, global_step=0)

for f in range(0,config.maximumFrames, config.numberEnvs):
    # Agent -- choose action
    actions = agents.select_actions( epsilon.get_epsilon(f), Network_Prediction, device, frameBuffers)

    # save current  state
    states = frameBuffers.return_stack()
    # move to next state
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
    Scores.addScore(rewards)
    Scores.AddTotalScoreAgent(dones,rewards)

    # train network
    _loss = 0
    _predicted_q = 0

    if replayBuffer.filled():
        for env in range(config.numberEnvs):
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
                """
                Because TD errors can be both positive (underestimated Q) and negative (overestimated Q), taking the plain .mean() will average them toward 0.0 even if individual errors are massive.
                Log the mean absolute TD error instead:
                """
                writer.add_scalar('tderror', td_error.abs().mean(), f)

            loss = criterion(predicted_q, target_q)
            loss.backward()
            optim.step()

            _loss += loss.detach()
            _predicted_q += predicted_q.mean().detach()

        # save data
        writer.add_scalar('loss', float(_loss)/config.numberEnvs, f)
        writer.add_scalar('epsilson', epsilon.get_epsilon(f), f)
        writer.add_scalar('AverageQValues', float(_predicted_q)/config.numberEnvs, f)
        writer.add_scalar('averageScoreRatio', Scores.ReturnAverageScores()["agent"], f)
        writer.add_scalar('averageScoreAgent', Scores.GetAverageTotalScores(config.numberEnvs), f)
        writer.add_scalar('updateTargetNetwork', 0, f)
        ## for later
        #writer.add_image('four_fashion_mnist_images', img_grid)
        # append to tensorboard
        if f % 100 ==0:
            writer.flush()
        if f % 5000==0:
            print("current iteration: {0} ".format( f  ) )
            print("  - epsilon: {0}".format(epsilon.get_epsilon(f)))
            print("  - average score ration agent/computer: {0}".format(Scores.ReturnAverageScores()["agent"]))

        # update target net
        if f % config.target_network_update_freq == 0:
            print("  - update target network !!")
            Network_Target.load_state_dict(Network_Prediction.state_dict())
            writer.add_scalar('updateTargetNetwork', 1, f)
        # replay buffer and netopw
        if f % 50000 == 0 and f > 1:
            print("  - saving replay buffer and model")
            replayBuffer.save_buffer(config.saveDir,  name = "replayBuffer_single")
            Networks.saveNetwork(
                network=Network_Prediction, saveDir=config.saveDir, iteration=f
            )




    """ 
    # save scores
    if f % 20 == 0 :
        Data.appendAverageScores()
    # print
    if f % 1000 == 0 and f > 1:
        print("current iteration: {0} with epsilon {1}".format( f ,  epsilon.get_epsilon(f)) )
        print("")
        Data.PrintAverageScores()
        Data.PrintAverageLosses(50)
        Data.PrintAverageQvalues(50)
        Data.PrintAverageTDerror(50)
        print("replay current size: {0}".format(replayBuffer.currentSize) )
        print("-------------------------------------")
    # save files
    """


