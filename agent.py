import Utilities
import numpy as np

import BufferVectorized
import gymnasium as gym
import Networks

import torch

class agents:
    def __init__(self, envs : gym.vector.SyncVectorEnv):
        self.epsilon = 1.0
        self.envs = envs

    def select_actions(self,epsilon : float, network_Prediction : Networks.DQN() ,device, frameBuffers : BufferVectorized.FrameBuffer()):
        self.epsilon = epsilon
        # -- random
        if np.random.random() < self.epsilon or not frameBuffers.filled():
            return  self.envs.action_space.sample()
        # -- predicted
        else:
            current_stack = frameBuffers.return_stack()
            with torch.no_grad():
                state = current_stack.float().to(device)
                q_values = network_Prediction(state)
                return q_values.argmax(dim=1).cpu().numpy()

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
        rampUnclamped =  min_out + (iteration - min_in) * (max_out - min_out) / (max_in - min_in)
        rampClamped =  min_out + (t_clamped - min_in) * (max_out - min_out) / (max_in - min_in)

        return rampClamped


