import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


class PongActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # actions: 0: NOOP, 2: RIGHT (UP), 3: LEFT (DOWN)
        self.action_mapping = [1, 2, 3]
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))

    def action(self, action):
        # Maps agent's action
        return self.action_mapping[action]


def make_env(seed):
    def thunk():
        env = gym.make('ALE/Pong-v5', frameskip=4)
        env = PongActionWrapper(env)
        env.action_space.seed(seed)
        return env
    return thunk

"""
def make_env(seed):
    def thunk():
        #with RGB
        render_mode = "human" if seed == 0 else None
        #env = gym.make('ALE/Pong-v5',frameskip=4,render_mode=render_mode)

        env = gym.make('ALE/Pong-v5', frameskip=4)
        # Apply the wrapper here
        env = PongActionWrapper(env)
        env.action_space.seed(seed)
        return env
    return thunk
"""