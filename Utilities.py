from collections import deque

import matplotlib.pyplot as plt

import statistics

from pathlib import Path


def plotState(state):
    plt.imshow(state)
    plt.title("3D NumPy Array Visualized as RGB Image (120 x 280 x 3)")
    plt.axis("off")
    plt.show()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

class CollectData:
    def __init__(self, ScoresDequeLength = 1000):
        # from game replay
        self.scoresDeque = {"agent": deque(maxlen=ScoresDequeLength), "computer": deque(maxlen=ScoresDequeLength)}
        self.ScoresAverages = {"agent": [], "computer": []}

        # from model prediction
        self.epsilons = []
        self.losses = []
        self.averageQlosses = []
        self.tdErrors = []

    def addScore(self, rewards ):
        # score vector
        for i,reward in enumerate(rewards):
            if reward != 0:
                if reward == 1:
                    self.scoresDeque["agent"].append(1)
                    self.scoresDeque["computer"].append(0)
                else:
                    self.scoresDeque["agent"].append(0)
                    self.scoresDeque["computer"].append(1)

    def appendAverageScores(self):
        if len(self.scoresDeque["agent"]) >10:
            AgentScores = statistics.mean(self.scoresDeque["agent"])
            ComputerScores = statistics.mean(self.scoresDeque["computer"])

            self.ScoresAverages["agent"].append(AgentScores)
            self.ScoresAverages["computer"].append(ComputerScores)

    def PrintAverageScores(self):
        if self.ScoresAverages["agent"] != [] and self.ScoresAverages["agent"] != []:
            print("Average average scores: Agent {0} || Computer {1}".format(
                self.ScoresAverages["agent"][-1],
                self.ScoresAverages["computer"][-1]) )

    # network data
    def addLosses(self, loss):
        self.losses.append(loss)

    def addTDError(self,value):
        self.tdErrors.append(value)

    def addAverageQValues(self, averageQvalue):
        self.averageQlosses.append(averageQvalue)

    def addEpsilon(self, epsilon):
        self.epsilons.append(epsilon)

    def PrintAverageLosses(self,maxLength=100):
        if len(self.losses) > maxLength:
            print("the current loss of the last {0} iteration is: {1}".format(maxLength, statistics.mean(self.losses[-maxLength:]) ))

    def PrintAverageQvalues(self,maxLength=100):
        if len(self.averageQlosses) > maxLength:
            print("the current average Qvalue of the last {0} iteration is: {1}".format(maxLength, statistics.mean(self.averageQlosses[-maxLength:]) ))

    def PrintAverageTDerror(self,maxLength=100):
        if len(self.tdErrors) > maxLength:
            print("the current average TD error of the last {0} iteration is: {1}".format(maxLength, statistics.mean(self.tdErrors[-maxLength:]) ))


    def save(self, directory):
        path = Path(directory) / "replayBuffer"
