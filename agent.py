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
                #device = Network_Prediction.get_device()
                state = current_stack.float().to(device) # we store as uint8 for efficiency hence scaling to [0,1] here directly
                q_values = network_Prediction(state)
                return q_values.argmax(dim=1).cpu().numpy()