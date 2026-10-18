from collections import deque
from typing import Deque

import matplotlib.pyplot as plt

import statistics
import types

from pathlib import Path


def plotState(state):
    plt.imshow(state)
    plt.title("3D NumPy Array Visualized as RGB Image (120 x 280 x 3)")
    plt.axis("off")
    plt.show()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

class Scores:
    def __init__(self, scoresDequeSize=1000):
        self.scoresDeque = {}
        self.scoresDeque["agent"] = Deque(maxlen=scoresDequeSize)
        self.scoresDeque["computer"] = Deque(maxlen=scoresDequeSize)
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
    def ReturnAverageScores(self):

        if len(self.scoresDeque["agent"]) >10:
            AgentScores = statistics.mean(self.scoresDeque["agent"])
            ComputerScores = statistics.mean(self.scoresDeque["computer"])
            return {"agent": AgentScores, "computer": ComputerScores}

        else:
            return {"agent": 0, "computer": 0}


def get_clean_hparams(module_or_dict):
    """Extracts non-dunder variables and keeps only float, int, or bool types."""
    source = (
        vars(module_or_dict)
        if not isinstance(module_or_dict, dict)
        else module_or_dict
    )

    # Inner helper function to validate allowed types
    def is_target_type(value):
        # isinstance(val, (int, float, bool)) checks for strictly int, float, or bool
        # (Note: in Python, bool is a subclass of int, so both are cleanly handled)
        return isinstance(value, (float, int, bool))

    # Filter out dunders, callables, modules, and keep only matching types
    filtered = {
        k: v
        for k, v in source.items()
        if not k.startswith("__")
        and not callable(v)
        and not isinstance(v, types.ModuleType)
        and is_target_type(v)
    }

    return filtered

"""
class CollectData:
    def __init__(self, ScoresDequeLength = 1000, configs={}):
        # from game replay
        self.scoresDeque = {"agent": deque(maxlen=ScoresDequeLength), "computer": deque(maxlen=ScoresDequeLength)}
        self.ScoresAverages = {"agent": [], "computer": []}

        # from model prediction
        self.epsilons = []
        self.losses = []
        self.averageQlosses = []
        self.tdErrors = []
        self.configs = configs

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
"""