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
    def __init__(self, rewardsDequeSize=1000, numberEnvs=4):
        self.rewardsDeque = {}
        self.rewardsDeque["agent"] = Deque(maxlen=rewardsDequeSize)
        self.rewardsDeque["computer"] = Deque(maxlen=rewardsDequeSize)
        self.rewardsDequeSize = rewardsDequeSize

        # per episode agent
        self.episodeScoresAgentBuffer = [0 for _ in range(numberEnvs)]
        self.totalEpisodeScoreAgent = []  # collect

        # length per episode
        self.episodeLengthBuffer = [0 for _ in range(numberEnvs)]
        self.episodeLengths = []

    # for average scores
    def addRewardsDeque(self, rewards):
        # score vector
        for i, reward in enumerate(rewards):
            if reward != 0:
                if reward == 1:
                    self.rewardsDeque["agent"].append(1)
                    self.rewardsDeque["computer"].append(0)
                else:
                    self.rewardsDeque["agent"].append(0)
                    self.rewardsDeque["computer"].append(1)

    def AverageScoreRatio(self):
        if len(self.rewardsDeque["agent"]) >= self.rewardsDequeSize:
            AgentScores = statistics.mean(self.rewardsDeque["agent"])
            return AgentScores
        else:
            return None

    # for Scores per episode
    def addEpisodeScoreAgent(self, dones, rewards, doPrint=False):
        # add reward to rewards buffer when agent scores
        indicesReward = np.where(rewards == 1)[0]
        for i in indicesReward:
            self.episodeScoresAgentBuffer[i] += 1

        # when episodes are done
        indicesDones = np.where(dones == 1)[0]
        if len(indicesDones) > 0:
            for i in indicesDones:
                self.totalEpisodeScoreAgent.append(self.episodeScoresAgentBuffer[i])
                if doPrint:
                    print(
                        "agent scored {1} points in Env: {0} !".format(
                            i, self.episodeScoresAgentBuffer[i]
                        )
                    )
                # reset buffer
                self.episodeScoresAgentBuffer[i] = 0

    def GetEpisodeAgent(self):
        if len(self.totalEpisodeScoreAgent):
            return self.totalEpisodeScoreAgent[-1]
        else:
            return -1

    def GetEpisodeAgentsWinAverage(self, avergeSize=10):
        if len(self.totalEpisodeScoreAgent) > avergeSize:
            npa = np.asarray(self.totalEpisodeScoreAgent, dtype=np.int8)
            npa = npa[-avergeSize:]
            wins = float(np.count_nonzero(npa == 21))
            return wins / avergeSize
        else:
            return 0

    def addEpisodeLengthBuffer(self, dones):
        for i, state in enumerate(dones):
            if state:
                self.episodeLengths.append(self.episodeLengthBuffer[i])
                self.episodeLengthBuffer[i] = 0
            else:
                self.episodeLengthBuffer[i] += 1

    def returnEpisodeLength(self, maxLength=8):
        if len(self.episodeLengths) >= maxLength:
            return statistics.mean(self.episodeLengths)
        else:
            return -1


def get_clean_hparams(module_or_dict):
    """Extracts non-dunder variables and keeps only float, int, or bool types."""
    source = (
        vars(module_or_dict) if not isinstance(module_or_dict, dict) else module_or_dict
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
