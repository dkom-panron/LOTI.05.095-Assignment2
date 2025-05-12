import sys
import numpy as np
import highway_env
import gymnasium as gym
from env_config import env_config

import time
import matplotlib.pyplot as plt

import argparse

from highway import EnvBarrierSim, set_obs

from planners.CEMPlanner import CEMPlanner

if __name__ == "__main__":
    env_config = env_config
    env = gym.make("highway-v0", config=env_config, render_mode="human")
    env.unwrapped.config["observation"]["absolute"] = True
    obs, info = env.reset(seed=121)

    lane_lower = env.unwrapped.road.network.graph["0"]["1"][+0]
    lane_upper = env.unwrapped.road.network.graph["0"]["1"][-1]

    lane_lb = lane_lower.start[1] - lane_lower.width/2
    lane_ub = lane_upper.start[1] + lane_upper.width/2

    parser = argparse.ArgumentParser(description="A program with different planner options")
    
    parser.add_argument(
        "--planner",
        type=str,
        choices=["cem", "nls", "ars"],
        default="cem",
        help="Planner type: 'cem', 'nls', or 'ars'"
    )
    
    args = parser.parse_args()

    num_controls = 20
    delta_t = 1/env.unwrapped.config["policy_frequency"]

    if args.planner == "cem":
        # TODO: commandline arguments
        planner = CEMPlanner(
            n=num_controls,
            num_samples=20,
            percentage_elite=0.1,
            num_iter=5,
            # calculate delta t based on env simulation step
            delta_t=delta_t,
            l=2.5,
            yd=8.0, # centerline y, each lane is 4 units wide
            vd=15.0, # desired speed
            min_v=env.unwrapped.config["action"]["speed_range"][0],
            max_v=env.unwrapped.config["action"]["speed_range"][1],
            min_steer=env.unwrapped.config["action"]["steering_range"][0],
            max_steer=env.unwrapped.config["action"]["steering_range"][1],
            beta=5.0
        )

        #mean_prev = np.zeros(2 * num_controls)
        mean_prev = None
    else:
        print("Planner not implemented yet")
        sys.exit(1)


    env_barrier = EnvBarrierSim(
        env.unwrapped.vehicle.WIDTH, 
        env.unwrapped.vehicle.LENGTH, 
        obs_count=env.unwrapped.config["observation"]["vehicles_count"] - 1,
        lane_count=env.unwrapped.config["lanes_count"],
        n=num_controls
    )


    ego_state, obs = set_obs(obs, lane_ub, lane_lb)
    done = False
    while not done:
        
        # Action to be computed using your Optimizer based on observation
        controls, mean_prev, action = planner.plan(ego_state, obs, mean=mean_prev)

        obs, reward, done, truncated, info = env.step(action)
        ego_state, obs = set_obs(obs, lane_ub, lane_lb)

        # Plot your generated trajectories here
        # TODO: Trajectory does not appear curved, rather staright
        x = np.cumsum(controls[:, 0] * np.cos(controls[:, 1]) * delta_t)
        y = np.cumsum(controls[:, 0] * np.sin(controls[:, 1]) * delta_t)
        env_barrier.lines[0].set_data(x, y)
        #print(f"{x=}")
        #print(f"{y=}")

        env_barrier.step(
            ego_state,  obs[1:], lane_lb, lane_ub
        )
        env.render()
