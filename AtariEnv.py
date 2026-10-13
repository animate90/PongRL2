import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

def make_env(rank: int):
    def _thunk():
        render_mode = "human" if rank == 0 else None
        g = gym.make("ALE/Pong-v5", render_mode=render_mode,frameskip=4)
        g.action_space.seed(rank*5)
        return g
    return _thunk