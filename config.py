import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



numberEnvs = 4 # number of virtual environments
FrameBufferSize = 4 # size of the state Stack
replayBufferCapacity = 800000 # max size of #300000 # tst with 50000 and 800000 for fun

# learning hyper paremater
learning_rate = 0.00025
discount_factor = 0.99
target_network_update_freq = 5000  # 500 standart  test 5 and 50000
batch_size = 128 # batch size for each training iteration

# epsilon behaviour
epsilon_initial = 1
epsilon_final = 0.125
epsilon_decay_Start = 10000
epsilon_decay_End = 200000
warmupiteration = 5000

maximumFrames = 1000000

doDoubleDQN = True

# data collection parameter
saveDir = 'D:/PongRL2'

# 100000   17-08-38  done
# 100000   22-49-49
# 200000   08-17-12  done
# 300000