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
from planners.JAXCEMPlanner import JAXCEMPlanner

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
    #planner = CEMPlanner(
    kwargs = {
      "n": num_controls,
      "num_samples": 200,
      "percentage_elite": 0.05,
      "num_iter": 50,
      # calculate delta t based on env simulation step
      "delta_t": delta_t,
      "l": 2.5,
      "yd": 8.0, # centerline y, each lane is 4 units wide
      "vd": 25.0, # desired speed
      "min_v": env.unwrapped.config["action"]["speed_range"][0],
      "max_v": env.unwrapped.config["action"]["speed_range"][1],
      "min_steer": env.unwrapped.config["action"]["steering_range"][0],
      "max_steer": env.unwrapped.config["action"]["steering_range"][1],
      "beta": 5.0
    }
    jax_cem_planner = JAXCEMPlanner(
      **kwargs,
    )

    cem_planner = CEMPlanner(
      **kwargs,
    )

    mean_prev = np.zeros(2 * num_controls)
    mean_prev[:num_controls] = jax_cem_planner.vd
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
  delta0_prev = 0.0

  # visualize controls
  fig, ax_controls = plt.subplots()
  ax_controls.set_title("Control Inputs Over Time")
  ax_controls.set_xlabel("Control Index")
  ax_controls.set_ylabel("Control Value")

  while not done:
    #print(f"ego_state: {ego_state}")
    print(f"obs: {obs}")

    action, v, steering, x_traj, y_traj, theta_traj, mean, controls, controls_best = cem_planner.plan(
      ego_state=ego_state, obs=obs, mean_init=mean_prev, delta0=delta0_prev
    )
    print("compute cost")
    cem_cost = cem_planner.compute_cost(
      controls_best, ego_state, obs, delta0_prev
    )
    jax_cem_cost = jax_cem_planner.compute_cost(
      controls_best, ego_state, obs, delta0_prev
    )
    print(f"cem_cost: {cem_cost}")
    print(f"jax_cem_cost: {jax_cem_cost}")
    break
    
    """
    # Action to be computed using your Optimizer based on observation
    #controls, mean_prev, action = planner.plan(ego_state, obs, mean=mean_prev)
    action, v, steering, x_traj, y_traj, theta_traj, mean, controls = jax_cem_planner.plan(
      ego_state, obs, mean_init=mean_prev, delta0=delta0_prev
    )

    # Update control plot
    plt.cla()
    # Plot controls from JAXCEMPlanner
    for c in controls:
      _x_traj, _y_traj, _, _, _ = jax_cem_planner.compute_rollout(c, ego_state, delta0_prev)
      ax_controls.plot(_x_traj - _x_traj[0], _y_traj - _y_traj[0], "b-", alpha=0.1)
    ax_controls.plot(x_traj - x_traj[0], y_traj - y_traj[0], "r-", label="JAX Best")

    # Plot controls from CEMPlanner
    _action, _v, _steering, _x_traj, _y_traj, _theta_traj, _mean, _controls = cem_planner.plan(
      ego_state, obs, mean_init=mean_prev, delta0=delta0_prev
    )
    for c in _controls:
      __x_traj, __y_traj, _, _, _ = cem_planner.compute_rollout(c, ego_state, delta0_prev)
      ax_controls.plot(__x_traj - __x_traj[0], __y_traj - __y_traj[0], "g-", alpha=0.1)
    ax_controls.plot(_x_traj - _x_traj[0], _y_traj - _y_traj[0], "y-", label="CEM Best")

    ax_controls.invert_yaxis()

    ax_controls.grid(True)
    ax_controls.legend()
    plt.pause(0.01)
    """


    mean_prev = mean
    delta0_prev = action[1]

    obs, reward, done, truncated, info = env.step(action)
    ego_state, obs = set_obs(obs, lane_ub, lane_lb)

    # Plot your generated trajectories here
    env_barrier.lines[0].set_data(x_traj - x_traj[0], y_traj - y_traj[0])
    #env_barrier.lines[0].set_data(_x_traj - _x_traj[0], _y_traj - _y_traj[0])
    #print(f"{x=}")
    #print(f"{y=}")

    env_barrier.step(
      ego_state,  obs[1:], lane_lb, lane_ub
    )


    env.render()

    input()
