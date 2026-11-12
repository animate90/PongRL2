import numpy as np

import BufferVectorized
import gymnasium as gym
import Networks

import torch

import Utilities

class agents:
    def __init__(self, envs : gym.vector.SyncVectorEnv):
        self.epsilon = 1.0
        self.envs = envs

    def select_actions(self,epsilon : float, network_Prediction : Networks.DQN ,device, frameBuffers : BufferVectorized.FrameBuffer, num_envs: int, currentFrame =0 ):
        self.epsilon = epsilon
        self.num_envs = num_envs

        current_stack = frameBuffers.return_stack()
        states = current_stack.float().div(255.0).to(device)

        with torch.no_grad():
            q_values = network_Prediction(states)
            greedy_actions =  q_values.argmax(dim=1).cpu().numpy()

        random_actions = self.envs.action_space.sample()
        explore_mask = np.random.random(self.num_envs) < epsilon

        actions = np.where(explore_mask, random_actions, greedy_actions)
        return actions

class epsilonDecay():
    def __init__(self, epsilon_initial = 1, epsilon_final = 0.1, epsilon_decay_Start = 25000,epsilon_decay_End = 200000):
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay_Start = epsilon_decay_Start
        self.epsilon_decay_End = epsilon_decay_End
    def get_epsilon(self, iteration):
        min_in = self.epsilon_decay_Start
        max_in = self.epsilon_decay_End
        min_out = self.epsilon_initial
        max_out = self.epsilon_final
        t_clamped = max(min(iteration, max_in), min_in)
        rampClamped =  min_out + (t_clamped - min_in) * (max_out - min_out) / (max_in - min_in)
        return rampClamped


