from collections import deque
import os
from pathlib import Path
import agent
import AtariEnv
import BufferVectorized
import config
import gymnasium as gym
import Networks
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import Utilities

# Ensure CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------
# 1. Hyperparameters & Settings
# -------------------------------------------------------------
N_STEP = 3
GAMMA_N = config.discount_factor**N_STEP  # 0.99^3 = 0.970299

# 2. Initialize Vectorized Environments
envs = gym.vector.SyncVectorEnv(
    [AtariEnv.make_env(i) for i in range(config.numberEnvs)]
)
action_space_size = (
    envs.action_space.nvec[0]
    if hasattr(envs.action_space, "nvec")
    else envs.single_action_space.n
)

# 3. Prime FrameBuffers
initial_obs, infos = envs.reset(seed=[100 + i for i in range(config.numberEnvs)])

frameBuffers = BufferVectorized.FrameBuffer(
    numberEnvs=config.numberEnvs, frameBufferSize=config.FrameBufferSize
)
for envID in range(config.numberEnvs):
    for _ in range(config.FrameBufferSize):
        frameBuffers.addFrameSingleEnv(envID, initial_obs[envID])

# 4. Initialize Replay Buffer & Per-Environment N-Step Queues
replayBuffer = BufferVectorized.ReplayBuffer(
    capacity=config.replayBufferCapacity,
    batch_size=config.batch_size,
    frameBufferSize=config.FrameBufferSize,
)
n_step_buffers = [deque(maxlen=N_STEP) for _ in range(config.numberEnvs)]

# 5. Initialize Networks (Double DQN)
Q_network = Networks.DQN(
    frameBufferSize=config.FrameBufferSize, actionSpaceSize=action_space_size
).to(device)
Q_network_Target = Networks.DQN(
    frameBufferSize=config.FrameBufferSize, actionSpaceSize=action_space_size
).to(device)
Q_network_Target.load_state_dict(Q_network.state_dict())
Q_network_Target.eval()

criterion = torch.nn.SmoothL1Loss()
optim = torch.optim.Adam(Q_network.parameters(), lr=config.learning_rate)

epsilon = agent.epsilonDecay(
    epsilon_initial=config.epsilon_initial,
    epsilon_final=config.epsilon_final,
    epsilon_decay_Start=config.epsilon_decay_Start,
    epsilon_decay_End=config.epsilon_decay_End,
)

agents = agent.agents(envs)
Scores = Utilities.Scores(numberEnvs=config.numberEnvs)
writer = SummaryWriter()


# -------------------------------------------------------------
# Helper: Extract N-Step Transition
# -------------------------------------------------------------
def get_n_step_transition(buffer_queue, gamma):
    """Calculates discounted sum of rewards across the queue and extracts (s_t, a_t, R_n, s_{t+n}, done)."""
    discounted_reward = sum(
        (gamma**i) * buffer_queue[i]["reward"] for i in range(len(buffer_queue))
    )
    s_0 = buffer_queue[0]["state"]
    a_0 = buffer_queue[0]["action"]
    s_n = buffer_queue[-1]["next_state"]
    done_n = buffer_queue[-1]["done"]
    return s_0, a_0, discounted_reward, s_n, done_n


# -------------------------------------------------------------
# Main Training Loop
# -------------------------------------------------------------
for f in range(0, config.maximumFrames):
    current_eps = epsilon.get_epsilon(f)

    # 1. Action selection
    actions = agents.select_actions(
        epsilon=current_eps,
        network_Prediction=Q_network,
        device=device,
        frameBuffers=frameBuffers,
        num_envs=config.numberEnvs,
    )

    # 2. Record state s_t
    states = frameBuffers.return_stack()

    # 3. Environment Step
    next_frames, rewards, terminateds, truncateds, infos = envs.step(actions)
    dones = np.logical_or(terminateds, truncateds)

    # 4. Advance FrameBuffer
    frameBuffers.addFrames(next_frames)
    next_states = frameBuffers.return_stack()

    # 5. Process Transitions through N-Step Queues
    for envID in range(config.numberEnvs):
        # Package 1-step interaction
        transition = {
            "state": states[envID].cpu().numpy(),
            "action": actions[envID],
            "reward": rewards[envID],
            "next_state": next_states[envID].cpu().numpy(),
            "done": dones[envID],
        }
        n_step_buffers[envID].append(transition)

        # Standard step: If queue is full, push the oldest N-step transition to ReplayBuffer
        if len(n_step_buffers[envID]) == N_STEP:
            s_0, a_0, r_n, s_n, d_n = get_n_step_transition(
                n_step_buffers[envID], config.discount_factor
            )
            replayBuffer.add(
                state=s_0, action=a_0, reward=r_n, next_state=s_n, done=d_n
            )

        # Terminal step: Flush all remaining transitions so terminal rewards (+1 / -1) are not lost
        if dones[envID]:
            while len(n_step_buffers[envID]) > 0:
                s_0, a_0, r_n, s_n, d_n = get_n_step_transition(
                    n_step_buffers[envID], config.discount_factor
                )
                replayBuffer.add(
                    state=s_0, action=a_0, reward=r_n, next_state=s_n, done=d_n
                )
                n_step_buffers[envID].popleft()

            # Re-prime deque for the new episode
            frameBuffers.reset_env(envID, next_frames[envID])

    # 6. Track Game Metrics
    Scores.addScore(rewards)
    Scores.AddTotalScoreAgent(dones, rewards)

    # 7. Optimize Network
    if replayBuffer.currentSize >= config.warmupiteration:
        batch = replayBuffer.sample()

        b_states = batch["states"].float().div(255.0)
        b_next_states = batch["next_states"].float().div(255.0)
        b_actions = batch["actions"].long().unsqueeze(1)
        b_rewards = batch["rewards"].float()
        b_dones = batch["dones"].float()

        with torch.no_grad():
            if config.doDoubleDQN:
                # Double DQN: Action selection via online net, evaluation via target net
                best_actions = (
                    Q_network(b_next_states).argmax(dim=1).unsqueeze(1)
                )
                next_q_values = (
                    Q_network_Target(b_next_states)
                    .gather(1, best_actions)
                    .squeeze(1)
                )
            else:
                next_q_values = (
                    Q_network_Target(b_next_states).max(dim=1).values
                )

            # Target update uses gamma^N
            target_y = b_rewards + (GAMMA_N * next_q_values * (1.0 - b_dones))

        current_q = Q_network(b_states).gather(1, b_actions).squeeze(1)

        loss = criterion(current_q, target_y)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(Q_network.parameters(), max_norm=10.0)
        optim.step()

        # 8. Target Network Synchronization
        if f % config.target_network_update_freq == 0:
            Q_network_Target.load_state_dict(Q_network.state_dict())
            writer.add_scalar("updateTargetNetwork", 1, f)

        # 9. Logging
        if f % 100 == 0:
            writer.add_scalar("loss", loss.item(), f)
            writer.add_scalar("AverageQValues", current_q.mean().item(), f)
            writer.add_scalar("epsilon", current_eps, f)
            writer.add_scalar(
                "averageScoreRatio",
                Scores.ReturnAverageScores()["agent"],
                f,
            )
            writer.add_scalar(
                "averageScoreAgent",
                Scores.GetAverageTotalScores(config.numberEnvs),
                f,
            )
            writer.flush()

    # Console Logging & Model Checkpointing
    if f % 5000 == 0 and f > 0:
        print(
            f"Iteration {f} | Epsilon: {current_eps:.3f} | Buffer: {replayBuffer.currentSize}"
        )
        print(f"  Score Ratio: {Scores.ReturnAverageScores()['agent']:.4f}")

    if f % 50000 == 0 and f > 0:
        Networks.saveNetwork(
            network=Q_network, saveDir=config.saveDir, iteration=f
        )