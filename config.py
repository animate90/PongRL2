import torch

numberEnvs = 8

FrameBufferSize = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128

replayBufferCapacity = 300000

# learning hyper paremater
learning_rate = 0.00025
discount_factor = 0.99
target_network_update_freq = 5000

# epsilon behaviour
epsilon_initial = 1
epsilon_final = 0.125
epsilon_decay_Start = 10000
epsilon_decay_End = 2000000

warmupiteration = 5000

maximumFrames = 1000000

doDoubleDQN = True

# data collection parameter
saveDir = 'D:/PongRL2'

