from collections import deque
from typing import Deque

import matplotlib.pyplot as plt

import statistics
import types

import numpy as np

from pathlib import Path


def plotState(state):
    plt.imshow(state)
    plt.title("3D NumPy Array Visualized as RGB Image (120 x 280 x 3)")
    plt.axis("off")
    plt.show()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

class Scores:
    def __init__(self, scoresDequeSize=1000, numberEnvs=4):
        self.scoresDeque = {}
        self.scoresDeque["agent"] = Deque(maxlen=scoresDequeSize)
        self.scoresDeque["computer"] = Deque(maxlen=scoresDequeSize)

        self.totalEpisodeScores = [] # collect
        self.EpisodeScoresBuffer = [0 for _ in range(numberEnvs)] # collect scores during epiosde per Env
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

    def AddTotalScoreAgent(self, dones, rewards, doPrint=False):
        indicesReward = np.where(rewards == 1)[0]
        #if len(indicesReward) > 0: print(indicesReward)
        for i in indicesReward:
            self.EpisodeScoresBuffer[i] += 1

        indicesDones = np.where(dones == 1)[0]
        if len(indicesDones) > 0:
            for i in indicesDones:
                self.totalEpisodeScores.append( self.EpisodeScoresBuffer[i])
                # reset buffer
                self.EpisodeScoresBuffer[i] = 0
                if doPrint:
                    print("agent score in Env: {0} is {0} !".format(i, rewards[indicesDones]) )


    def GetAverageTotalScores(self, maxLength=20):
        totalScore = 0
        counter = 0
        for i,score in enumerate(self.totalEpisodeScores):
            totalScore += score
            counter += 1
            if i > maxLength:
                break
        if counter >= 1:
            return float(totalScore) / float(counter)
        else:
            return 0




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

