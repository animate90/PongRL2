from torch import nn
from pathlib import Path

#from torch import save
import torch

class DQN(nn.Module):
    def __init__(self, frameBufferSize=4, actionSpaceSize=6):
        self.actionSpaceSize = actionSpaceSize
        self.frameBufferSize = frameBufferSize
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(self.frameBufferSize,32,8,4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136,512),
            nn.ReLU(),
            nn.Linear(512,self.actionSpaceSize)
        )
    def forward(self, x):
        return self.model(x)


def saveNetwork(network, saveDir, filename="model", iteration=None):
    path = Path(saveDir) / "modelWeights"
    path.mkdir(parents=True, exist_ok=True)
    # file name
    if iteration == None:
        path = path / (filename + ".pth")
    else:
        path = path / (filename + "__" + str(iteration) + ".pth" )

    torch.save(network.state_dict(), path )

def loadNetwork(saveDir, filename="model", iteration=None):
    print("not implemented")
