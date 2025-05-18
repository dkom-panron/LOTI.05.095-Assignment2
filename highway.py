import sys
import numpy as np
import highway_env
import gymnasium as gym
from env_config import env_config

import time
import matplotlib.pyplot as plt


class EnvBarrierSim:
    def __init__(self, width, height, obs_count, lane_count=4, num=100, figsize=(10, 4), vx=(-30, 80), vy=(-20, 20), n: int = 500):
        self.X, self.Y = np.meshgrid(np.linspace(*vx, n), np.linspace(*vy, n))
        self.points = np.stack([self.X, self.Y], axis=-1)

        self.LANE_COUNT = lane_count
        self.LANE_WIDTH = 4
        self.WIDTH, self.HEIGHT = height, width
        self.A = np.array([[1., 0.], [-1., 0.], [0., 1.], [0, -1.]])
        self.b = np.array([self.WIDTH/2, self.WIDTH/2, self.HEIGHT/2, self.HEIGHT/2])

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect("equal")
        self.ax.grid(visible=True, linewidth=0.2)

        self.ego = plt.Rectangle(
            [-self.WIDTH/2, -self.HEIGHT/2],
            width=self.WIDTH, height=self.HEIGHT, zorder=1,
            angle=0, rotation_point="center", facecolor="grey"
        )
        self.ax.add_artist(self.ego)

        self.lines = self.ax.plot(*[np.empty((0, 1)) for _ in range(2 * num)], lw=0.7, color="blue", alpha=0.1)
        self.best_line = self.ax.plot(*[np.empty((0, 1)) for _ in range(2)], lw=1.0, color="red", alpha=1.0)
        self.lanes = self.ax.plot(*[np.empty((0, 1)) for _ in range(2 * (lane_count + 1))], lw=0.7)

        obstacles = np.zeros((obs_count, 5))
        obs_color="blue"
        self.obstacles = [
            plt.Rectangle(
                [center[0] - self.WIDTH/2, center[1] - self.HEIGHT/2],
                width=self.WIDTH, height=self.HEIGHT,
                angle=np.rad2deg(center[2]), rotation_point="center", facecolor=obs_color
            )
            for center in obstacles
        ]
        for ob in self.obstacles:
            self.ax.add_artist(ob)

        plt.axis([*vx, *vy])
        self.ax.set_ylim(self.ax.get_ylim()[::-1])

    def step(self, ego_state, obs_state, lane_lb, lane_ub):
        self.ego.set(angle=np.rad2deg(ego_state[4]))

        for (rec, obs) in zip(self.obstacles, obs_state):
            rec.set(
                xy=[obs[0] - self.WIDTH/2, obs[1] - self.HEIGHT/2],
                angle=np.rad2deg(obs[-1])
            )

        x_lane = np.arange(-100, 100)
        for i in range(1+self.LANE_COUNT):
            y_lane = (lane_lb + 4*i - ego_state[1]) * np.ones_like(x_lane)
            self.lanes[i].set_data(x_lane, y_lane)
            self.lanes[i].set(zorder=0)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def set_obs(obs, global_lane_up, global_lane_lb):
    ego_state = np.copy(obs[0])

    lb_lane = global_lane_lb - obs[0][1]
    ub_lane = global_lane_up - obs[0][1]

    ego_pose = obs[0, :2]
    obs[:, :2] = obs[:, :2] - ego_pose 
    obs[0, :2] = np.asarray([lb_lane, ub_lane])

    return ego_state, obs


if __name__ == "__main__":
    env_config = env_config
    env = gym.make("highway-v0", config=env_config, render_mode="human")
    env.unwrapped.config["observation"]["absolute"] = True
    obs, info = env.reset(seed=121)

    lane_lower = env.unwrapped.road.network.graph["0"]["1"][+0]
    lane_upper = env.unwrapped.road.network.graph["0"]["1"][-1]

    lane_lb = lane_lower.start[1] - lane_lower.width/2
    lane_ub = lane_upper.start[1] + lane_upper.width/2

    env_barrier = EnvBarrierSim(
        env.unwrapped.vehicle.WIDTH, 
        env.unwrapped.vehicle.LENGTH, 
        obs_count=env.unwrapped.config["observation"]["vehicles_count"] - 1,
        lane_count=env.unwrapped.config["lanes_count"]
    )

    ego_state, obs = set_obs(obs, lane_ub, lane_lb)
    done = False
    while not done:
        
        # Action to be computed using your Optimizer based on observation
        action = [0, 0]     

        obs, reward, done, truncated, info = env.step(action)
        ego_state, obs = set_obs(obs, lane_ub, lane_lb)

        # Plot your generated trajectories here
        env_barrier.lines[0].set_data(np.arange(100), np.arange(100) * 0)

        env_barrier.step(
            ego_state,  obs[1:], lane_lb, lane_ub
        )
        env.render()
        
