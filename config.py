import torch


numberEnvs = 8

FrameBufferSize = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

replayBufferCapacity = 1000000

# learning hyper paremater
learning_rate = 0.00025
discount_factor = 0.99
target_network_update_freq = 10000

epsilon_initial = 1
epsilon_final = 0.1

epsilon_decay_Start = 25000
epsilon_decay_End = 400000


virtualEnvsTrainingMultiplier = .5

maximumFrames = 1000000


# data collection parameter
saveDir = 'D:/PongRL2'

