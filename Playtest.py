import gymnasium as gym
import ale_py

import Networks
import config
import agent
import BufferVectorized

import torch

import numpy as np

from pathlib import Path

gym.register_envs(ale_py)


env = gym.make("ALE/Pong-v5", frameskip=4, render_mode="human")
action_space_size = 6
initial_obs, infos = env.reset()


# init agents
agents = agent.agents([env])
epsilon = agent.epsilonDecay(
    config.epsilon_initial,
    config.epsilon_final,
    config.epsilon_decay_Start,
    config.epsilon_decay_End,
)

# load Q function
# initialize Deep Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
online_network = Networks.DQN(config.FrameBufferSize, action_space_size).to(device)
# load old weights
path = Path(r"/modelWeights_DoubleDQN_Oct22_22-14-55_animate") / "model__5000.pth"
online_network.load_state_dict(torch.load(path, weights_only=True))


# initilize Buffers
frameBuffers = BufferVectorized.FrameBuffer(1, config.FrameBufferSize)


for f in range(0, config.maximumFrames):

    current_stack = frameBuffers.return_stack()
    states = current_stack.float().div(255.0).to(device)

    q_values = online_network(states)
    greedy_actions = q_values.argmax(dim=1).cpu().numpy()[0]

    # save current  state
    states = frameBuffers.return_stack()
    # move to next state
    next_obs, rewards, terminateds, truncateds, infos = env.step(greedy_actions)

    next_obs = np.expand_dims(next_obs, axis=0)

    frameBuffers.addFrames(next_obs)
