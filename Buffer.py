import cv2
from collections import deque
import torch
import numpy as np
import matplotlib.pyplot as plt
import config



actionSpace = 6



class FrameBuffer(object):
    def __init__(self,  frameBufferSize=4):
        self.frameBufferSize = frameBufferSize
        self.frames = deque(maxlen=frameBufferSize)

    def preprocess_frame_single(self, input_frame):
        frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)  # to grea
        frame = frame[30:,:] # crop scores
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) # resize
        #frame = frame.astype(np.float32) / 255.0
        frame = frame.astype(np.float32) / 255.0
        return frame

    def addFrame(self, input_frame):
        frame = self.preprocess_frame_single(input_frame)
        self.frames.append(frame)

    def return_stack(self):
        stack = np.stack(self.frames)
        return torch.from_numpy(stack).float()

    def filled(self):
        return self.frameBufferSize == len(self.frames)

    def display_stack(self):
        stack = self.return_stack()
        plt.imshow(torch.hstack([x.squeeze() for x in stack.split(1)]),cmap='gray')
        plt.axis('off')
        plt.show()
        plt.close()

class ReplayBuffer:
    def __init__(self, capacity =10000, batch_size=64,frameBufferSize=4):
        self.capacity = capacity
        self.device = config.device
        self.batch_size = batch_size
        self.index = 0
        self.frameBufferSize = frameBufferSize
        self.currentSize = 0

        self.states = np.zeros((self.capacity, self.frameBufferSize, 84, 84), dtype=np.float32)  # 4 stacked frames
        self.next_states = np.zeros((self.capacity, self.frameBufferSize, 84, 84), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.currentSize = min(self.currentSize + 1, self.capacity)

    def sample(self):
        self.batch_size = min(self.currentSize, self.batch_size)
        indices = np.random.randint(0, self.currentSize, self.batch_size)
        return {
            'states': torch.from_numpy(self.states[indices]).to(self.device),
            'actions': torch.from_numpy(self.actions[indices]).to(self.device),
            'rewards': torch.from_numpy(self.rewards[indices]).to(self.device),
            'next_states': torch.from_numpy(self.next_states[indices]).to(self.device),
            'dones': torch.from_numpy(self.dones[indices]).to(self.device)
        }

    def filled(self):
        return self.currentSize > self.batch_size

