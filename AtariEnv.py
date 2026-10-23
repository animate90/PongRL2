import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


class PongActionWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        # Real ALE actions: 0: NOOP, 2: RIGHT (UP), 3: LEFT (DOWN)
        self.action_mapping = [0, 2, 3]
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))


def make_env(seed):
    def thunk():
        env = gym.make('ALE/Pong-v5',frameskip=4)
        env.action_space.seed(seed)
        return env
    return thunk

"""
def make_env(rank: int):
    def _thunk():
        #render_mode = "human" if rank == 0 else None
        #g = gym.make("ALE/Pong-v5", render_mode=render_mode,frameskip=4)
        g = gym.make("ALE/Pong-v5", frameskip=4)
        g.action_space.seed(rank*5)


        g = PongActionWrapper(g)

        return g
    return _thunk
"""