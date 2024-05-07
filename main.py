import gymnasium as gym
import time
from env_config import env_config


if __name__ == "__main__":
    env_config = env_config
    env = gym.make("highway-v0", config=env_config, render_mode="human")
    env.reset(seed=121)

    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = [0, 0]
        obs, reward, done, truncated, info = env.step(action)