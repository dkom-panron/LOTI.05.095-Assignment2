import gymnasium as gym
from env_config import env_config


if __name__ == "__main__":
    env = gym.make('highway-v0', render_mode='human')

    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = 0
        obs, reward, done, truncated, info = env.step(action)