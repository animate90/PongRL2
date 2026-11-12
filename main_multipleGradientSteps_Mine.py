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

# adjust for vectorized envs
config.epsilon_decay_Start = int(config.epsilon_decay_Start/ config.numberEnvs)
config.epsilon_decay_End = int(config.epsilon_decay_End/ config.numberEnvs)
config.target_network_update_freq = int( config.target_network_update_freq  / config.numberEnvs)
config.warmupiteration = int( config.warmupiteration  / config.numberEnvs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Create our training environmentS
envs = gym.vector.SyncVectorEnv([AtariEnv.make_env(i) for i in range(config.numberEnvs)])
action_space_size = envs.action_space.nvec[0]
initial_obs, infos = envs.reset(seed=[100 + i for i in range(config.numberEnvs)])

# initilize Buffers
frameBuffers = BufferVectorized.FrameBuffer(config.numberEnvs , config.FrameBufferSize)
# prime buffer
for _ in range(config.FrameBufferSize):
    frameBuffers.addFrames( initial_obs)

prev_dones = np.zeros(config.numberEnvs, dtype=bool)

replayBuffer = BufferVectorized.ReplayBuffer(
    capacity=config.replayBufferCapacity,
    frameBufferSize=config.FrameBufferSize,
    batch_size=config.batch_size
)

# initialize Deep Network
online_network = Networks.DQN(config.FrameBufferSize, action_space_size).to(device)
network_Target = Networks.DQN(config.FrameBufferSize, action_space_size).to(device)
network_Target.load_state_dict(online_network.state_dict())

criterion = torch.nn.SmoothL1Loss() # Human-level control through deep reinforcement learning
optim = torch.optim.Adam(online_network.parameters(), lr=config.learning_rate)

# init agents
agents = agent.agents(envs)
epsilon = agent.epsilonDecay(config.epsilon_initial, config.epsilon_final, config.epsilon_decay_Start,config.epsilon_decay_End)

# Training
# -------------------------------------------------------------
# metrics
Scores = Utilities.Scores(rewardsDequeSize=250,numberEnvs=config.numberEnvs)
writer = SummaryWriter()


hparams = Utilities.get_clean_hparams(config)
result = "\n".join(f"{k}: {v}" for k, v in hparams.items())
writer.add_text("Hyperparameter", result, global_step=0)

for f in range(0,config.maximumFrames):
    # Agent -- choose action
    actions = agents.select_actions( epsilon.get_epsilon(f), online_network, device, frameBuffers, config.numberEnvs,f)

    # save current  state
    states = frameBuffers.return_stack()
    # move to next state
    next_obs, rewards, terminateds,truncateds, infos = envs.step(actions)
    dones = np.logical_or(terminateds, truncateds)
    # add next state to frame buffer
    frameBuffers.addFrames(next_obs)
    next_stack = frameBuffers.return_stack()

    for envID in range(config.numberEnvs):
        if prev_dones[envID]:
            # this step was the autoreset: next_obs is the first frame of a new episode
            for _ in range(config.FrameBufferSize):
                frameBuffers.addFrameSingleEnv(envID, next_obs[envID])
            continue  # don't store this transition
        replayBuffer.add(states[envID], actions[envID], rewards[envID],
                         next_stack[envID], terminateds[envID])
    prev_dones = dones
    # score vector
    Scores.addRewardsDeque(rewards)
    Scores.addEpisodeScoreAgent(dones,rewards,doPrint=False)
    Scores.addEpisodeLengthBuffer(dones)


    # train network
    _loss = 0
    _predicted_q = 0
    _td_errors = 0

    if replayBuffer.filled() and replayBuffer.currentSize > config.warmupiteration:
        for env in range(config.numberEnvs):
            # sample random Mni batches
            Buffer = replayBuffer.sample()
            state_input = Buffer["states"].float().div(255).to(device)
            next_state_input = Buffer["next_states"].float().div(255).to(device)
            rewards = Buffer["rewards"].view(-1)
            actions = Buffer["actions"].view(-1, 1).to(device)
            dones = Buffer["dones"].view(-1).float().to(device)

            # DQN
            if not config.doDoubleDQN:
                with torch.no_grad():
                    Y = rewards
                    Y += config.discount_factor * network_Target(next_state_input).max(dim=1).values * (1-dones)
            # double DQN
            else:
                with torch.no_grad():
                    max_a = online_network(next_state_input).argmax(dim=1).unsqueeze(1)
                    Y = rewards + config.discount_factor * network_Target(next_state_input).gather(1,max_a).squeeze(1) * (1 - dones)

            Q = online_network(state_input).gather(1, actions).view(-1)
            loss = criterion( Y, Q)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
            optim.step()


            # for data collection
            _loss += loss.mean().detach()
            _predicted_q += Q.mean().detach()
            _td_errors = (Y - Q).mean().detach()


        # save data to tensorboard
        writer.add_scalar('loss', _loss, f)
        writer.add_scalar('epsilson', epsilon.get_epsilon(f), f)
        writer.add_scalar("Td error", _td_errors ,f)
        writer.add_scalar('Average QValues', float(_predicted_q)/config.numberEnvs, f)
        if Scores.AverageScoreRatio() is not None:
            writer.add_scalar('Average ScoreRatio', Scores.AverageScoreRatio(), f)
        writer.add_scalar('Get EpisodeAgent Score', Scores.GetEpisodeAgent(), f)
        writer.add_scalar("Episode Agents Wins (Average)", Scores.GetEpisodeAgentsWinAverage(),f)
        writer.add_scalar("Average EpisodeLength", Scores.returnEpisodeLength(),f)


        # append to tensorboard
        if f % 100 ==0:
            writer.flush()
        if f % 5000==0:
            print("current iteration: {0} ".format( f  ) )
            print("  - epsilon: {0}".format(epsilon.get_epsilon(f)))
            if Scores.AverageScoreRatio() is not None:
                print("  - average score ration agent/computer: {0}".format(Scores.AverageScoreRatio()))
            print("  - replayBuffer Size: {0}".format(replayBuffer.currentSize))

        # update target net
        if f % config.target_network_update_freq == 0:
            #print("current iteration: {0} ".format( f  ) )
            #print("  - update target network !!")
            network_Target.load_state_dict(online_network.state_dict())
            writer.add_scalar('updateTargetNetwork', 1, f)
        else:
            writer.add_scalar('updateTargetNetwork', 0, f)
        # replay buffer and netopw
        if f % 5000 == 0 and f > 1:
            print("current iteration: {0} ".format( f  ) )
            print("  - saving replay buffer and model")
            #replayBuffer.save_buffer(config.saveDir,  name = "replayBuffer_single")
            Networks.saveNetwork(
                network=online_network, saveDir=config.saveDir, iteration=f
            )



