import gymnasium as gym
import minigrid

def make_env(env_key, seed=None, render_mode=None, wrappers=None):
    env = gym.make(env_key, render_mode=render_mode)
    if wrappers is not None:
        for wrapper in wrappers:
            env = wrapper(env)
    env.reset(seed=seed)
    return env
