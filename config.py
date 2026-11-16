import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maximumFrames = 1000000

numberEnvs = 4
FrameBufferSize = 4
replayBufferCapacity = 800000

# learning hyper paramater
learning_rate = 0.00025
discount_factor = 0.99
target_network_update_freq = 500
batch_size = 128

# epsilon behaviour
epsilon_initial = 1
epsilon_final = 0.125
epsilon_decay_Start = 10000
epsilon_decay_End = 200000
warmupiteration = 5000

doDoubleDQN = True

# for data collection data collection parameter
saveDir = "D:/PongRL2"
