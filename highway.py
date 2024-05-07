import sys
import numpy as np
import gymnasium as gym
from env_config import env_config

import time
import torch as th
import matplotlib.pyplot as plt


def rotation_matrix(theta):
    return th.stack([
        th.stack([th.cos(theta), -th.sin(theta)], dim=-1),
        th.stack([th.sin(theta),  th.cos(theta)], dim=-1)
    ], dim=-2).squeeze()

class EnvBarrierSim:
    def __init__(self, width, height, lane_count=4, figsize=(10, 4), vx=(-30, 80), vy=(-20, 20), n: int = 500):
        self.X, self.Y = th.meshgrid(th.linspace(*vx, n), th.linspace(*vy, n), indexing="ij")
        self.points = th.stack([self.X, self.Y], dim=-1)

        self.LANE_COUNT = lane_count
        self.LANE_WIDTH = 4
        self.WIDTH, self.HEIGHT = height, width
        self.A = th.tensor([[1., 0.], [-1., 0.], [0., 1.], [0, -1.]])
        self.b = th.tensor([self.WIDTH/2, self.WIDTH/2, self.HEIGHT/2, self.HEIGHT/2])

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

        self.lanes = self.ax.plot(*[np.empty((0, 1)) for _ in range(2 * 5)], lw=0.7)

        plt.axis([*vx, *vy])
        self.ax.set_ylim(self.ax.get_ylim()[::-1])

    def step(self, ego_state, obs_state, lane_lb, lane_ub):
        self.ego.set(angle=np.rad2deg(ego_state[4]))

        obs_pos = obs_state[:, :2][:, None, None, :]
        points = self.points[None, ...] - obs_pos

        obs_ang = obs_state[:, 4]
        obs_mat = rotation_matrix(obs_ang) @ self.A.mT
        A_rot = th.func.vmap(lambda x, y: x@y)(points, obs_mat)
        h, _ = (self.b - A_rot).min(dim=-1)
        Z = - th.log(-h/(1-h))
        Z = Z.sum(dim=0)

        for coll in plt.gca().collections:
            coll.remove()

        self.ax.contour(self.X, self.Y, Z, 100, linewidths=1.0, cmap="Purples", zorder=-1)

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
    ub_lane = global_lane_up + obs[0][1]

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
        lane_count=env_config["lanes_count"]
    )

    ego_state, obs = set_obs(obs, lane_ub, lane_lb)
    for _ in range(500):
        action = [0, 0]     # Optimizer(obs)
        obs, reward, done, truncated, info = env.step(action)
        ego_state, obs = set_obs(obs, lane_ub, lane_lb)

        print(obs)
        env_barrier.step(
            th.from_numpy(ego_state), 
            th.from_numpy(obs[1:]), 
            lane_lb, lane_ub
        )
        time.sleep(1000)
        env.render()
        