import torch


numberEnvs = 12

FrameBufferSize = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

replayBufferCapacity = 1000000


learning_rate = 0.00025
discount_factor = 0.99

target_network_update_freq = 10000

initial_epsilon = 1
final_epsilon = 0.1
final_exploration_iteration = 1000000


saveDir = 'D:/PongRL2'