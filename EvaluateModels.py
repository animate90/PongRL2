from pathlib import Path

import gymnasium as gym
import torch
import numpy as np

import csv

import AtariEnv
import agent
import config
import Networks
import BufferVectorized

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config.numberEnvs = 1

# find model weight
models = {}
mainPath = Path(r"D:\PongRL2")
for f in mainPath.iterdir():
    if f.name.startswith("modelWeights__"):
        files = []
        for file in f.iterdir():
            files.append(file)
        models[f.name] = files

# start env
# 1. Create our training environmentS
envs = gym.vector.SyncVectorEnv(
    [AtariEnv.make_env(i) for i in range(config.numberEnvs)]
)
action_space_size = envs.action_space.nvec[0]
initial_obs, infos = envs.reset(seed=[100 + i for i in range(config.numberEnvs)])
# init agents
agents = agent.agents(envs)
# initialize Deep Network
online_network = Networks.DQN(config.FrameBufferSize, action_space_size).to(device)
# initilize Buffers
frameBuffers = BufferVectorized.FrameBuffer(config.numberEnvs, config.FrameBufferSize)
# prime buffer
for _ in range(config.FrameBufferSize):
    frameBuffers.addFrames(initial_obs)


# number of episodes per models:
maxEpisodes = 10

Scores = {}

for model in models.keys():
    Scores = {}
    print(model)
    for weight in models[model]:
        iteration = weight.name[7:-4]

        print("evaluation of model: {0}".format(weight))
        online_network.load_state_dict(torch.load(weight))

        rewards_coll = np.zeros(config.numberEnvs)
        episodeScores = []
        EpisodeCounter = 0

        while EpisodeCounter <= maxEpisodes:

            # pure exploitation
            actions = agents.select_actions(
                0.05, online_network, device, frameBuffers, config.numberEnvs
            )

            # save current  state
            states = frameBuffers.return_stack()
            # move to next state
            next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            # add next state to frame buffer
            frameBuffers.addFrames(next_obs)

            # 1. Accumulate TRUE rewards for score tracking (-21 to +21)
            # Flatten to 1D to prevent dimension broadcasting bugs
            rewards_flat = np.asarray(rewards).reshape(-1)
            dones_flat = np.asarray(dones).reshape(-1)

            rewards_coll += rewards_flat

            # 2. Extract done indices properly using [0] or np.flatnonzero
            if np.any(dones_flat):
                done_indices = np.flatnonzero(dones_flat)

                for i in done_indices:
                    score = float(rewards_coll[i])
                    episodeScores.append(score)
                    rewards_coll[i] = 0.0
                    EpisodeCounter += 1

                # print(f"Recorded scores: {episodeScores[-len(done_indices):]}")
                print(episodeScores)

            # 3. Log episode scores at checkpoint intervals
            Scores[iteration] = list(episodeScores)

    print(Scores)
    print("write CSV")

    with open(
        mainPath / (model + "PlayedPongScores.csv"),
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        # Optional header
        # writer.writerow(["Key", "Item", "Color", "Quantity"])

        for key, values in Scores.items():
            # Writes the key as column 1, followed by each item in the list
            aver = float(np.average(values))
            writer.writerow([key, aver, *values])
