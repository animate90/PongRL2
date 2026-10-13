import cv2
from collections import deque
import torch
import numpy as np
import matplotlib.pyplot as plt
import config

from pathlib import Path


class FrameBuffer(object):
    def __init__(self, numberEnvs = 2, frameBufferSize=4):
        self.numberEnvs = numberEnvs
        self.frameBufferSize = frameBufferSize
        self.frames = [deque(maxlen=frameBufferSize) for _ in range(numberEnvs)]

    def preprocess_frame_single(self, input_frame):
        frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)  # to grea
        frame = frame[30:,:] # crop scores
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) # resize
        #frame = frame.astype(np.float32) / 255.0
        frame = frame.astype(np.int8)
        return frame

    def addFrames(self, input_frames):
        for index in range(self.numberEnvs):
            frame = self.preprocess_frame_single(input_frames[index])
            self.frames[index].append(frame)

    def return_stack(self):
        stackFrameBuffer = []
        for envIndex in range(self.numberEnvs):
            stackFrameBuffer.append(np.stack(self.frames[envIndex]))
        stack = np.stack(stackFrameBuffer)
        return torch.from_numpy(stack).float()

    def filled(self):
        # make it fail save to go through each env
        return self.frameBufferSize == len(self.frames[0])

    def display_stack(self, envIndex=0):
        stack = self.return_stack()[envIndex]
        plt.imshow(torch.hstack([x.squeeze() for x in stack.split(1)]),cmap='gray')
        plt.axis('off')
        plt.title(f'env {envIndex}')
        plt.show()
        plt.close()

    def display_stacks(self):
        stack = self.return_stack()
        tiled_image = stack.permute(0, 2, 1, 3).reshape(stack.shape[0] * 84, stack.shape[1] * 84)
        plt.figure(figsize=(12, 12))
        plt.imshow(tiled_image, cmap="gray")
        plt.axis("off")  # Hides axis ticks and borders
        plt.show()

class ReplayBuffer:
    def __init__(self, capacity = 10000, batch_size=64,frameBufferSize=4):
        self.capacity = capacity
        self.device = config.device
        self.batch_size = batch_size
        self.index = 0
        self.frameBufferSize = frameBufferSize
        self.currentSize = 0

        self.states = np.zeros((self.capacity, self.frameBufferSize, 84, 84), dtype=np.int8)  # 4 stacked frames
        self.next_states = np.zeros((self.capacity, self.frameBufferSize, 84, 84), dtype=np.int8)
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

    def save_buffer(self, directory):
        path = Path(directory) / "replayBuffer"
        arrays = ["states", "actions", "rewards", "next_states", "dones"]
        for array in arrays:
            p = path / array
            p.mkdir(parents=True, exist_ok=True)
            value = getattr(self, array)
            filename = array + ".npy"
            np.savez_compressed(p / filename, value)
            #np.save(p / filename, value)