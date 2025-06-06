import sys
import numpy as np
import highway_env
import gymnasium as gym
from env_config import env_config
import jax
import jax.numpy as jnp

import time
import matplotlib.pyplot as plt

from assignment_args import parse_args

from highway import EnvBarrierSim, set_obs

from planners.CEMPlanner import CEMPlanner
from planners.JAXCEMPlanner import JAXCEMPlanner
from planners.NLSPlanner import NLSPlanner
from planners.ARSPlanner import ARSPlanner

if __name__ == "__main__":
  # NB! Very important.
  # When calculating lane cost using softplus, single precision
  # can lead to numerical issues.
  jax.config.update("jax_enable_x64", True)
    
  args = parse_args()

  env_config = env_config
  # Update env_config with command line arguments
  env_config["initial_lane_id"] = args.env_initial_lane_id

  env = gym.make("highway-v0", config=env_config, render_mode="human")
  env.unwrapped.config["observation"]["absolute"] = True
  obs, info = env.reset(seed=121)

  lane_lower = env.unwrapped.road.network.graph["0"]["1"][+0]
  lane_upper = env.unwrapped.road.network.graph["0"]["1"][-1]

  lane_lb = lane_lower.start[1] - lane_lower.width/2
  lane_ub = lane_upper.start[1] + lane_upper.width/2


  delta_t = 1/env.unwrapped.config["policy_frequency"]

  if args.planner == "cem":
    kwargs = {
      "n": args.n,
      "num_samples": args.cem_samples,
      "percentage_elite": args.cem_elite,
      "num_iter": args.cem_iter,
      # calculate delta t based on env simulation step
      "delta_t": delta_t,
      "l": 2.5,
      "yd": args.goal_yd,
      "vd": args.goal_vd,
      "min_v": env.unwrapped.config["action"]["speed_range"][0],
      "max_v": env.unwrapped.config["action"]["speed_range"][1],
      "min_steer": env.unwrapped.config["action"]["steering_range"][0],
      "max_steer": env.unwrapped.config["action"]["steering_range"][1],
      "stomp_like": args.cem_stomp_like,
    }
    planner = JAXCEMPlanner(
      **kwargs,
    )

    #cem_planner = CEMPlanner(
    #  **kwargs,
    #)

    mean_prev = np.zeros(2 * args.n)
    # NB! Not initializing initial velocities to goal velocity
    # led to the very first CEM samples being all over the place.
    mean_prev[:args.n] = planner.vd
  elif args.planner == "nls":
    kwargs = {
      "n": args.n,
      "num_iter": args.nls_iter,
      "delta_t": delta_t,
      "l": 2.5,
      "yd": args.goal_yd,
      "vd": args.goal_vd,
      "min_v": env.unwrapped.config["action"]["speed_range"][0],
      "max_v": env.unwrapped.config["action"]["speed_range"][1],
      "min_steer": env.unwrapped.config["action"]["steering_range"][0],
      "max_steer": env.unwrapped.config["action"]["steering_range"][1],
    }
    planner = NLSPlanner(
      **kwargs,
    )

    controls_prev = 0.01 * jnp.ones(2 * args.n)
    #controls_prev = jnp.zeros(2 * args.n)
    controls_prev = controls_prev.at[:args.n].set(planner.vd)
  elif args.planner == "ars":
    kwargs = {
      "n": args.n,
      "num_samples": args.ars_samples,
      "percentage_elite": args.ars_elite,
      "num_iter": args.ars_iter,
      # calculate delta t based on env simulation step
      "delta_t": delta_t,
      "l": 2.5,
      "yd": args.goal_yd,
      "vd": args.goal_vd,
      "min_v": env.unwrapped.config["action"]["speed_range"][0],
      "max_v": env.unwrapped.config["action"]["speed_range"][1],
      "min_steer": env.unwrapped.config["action"]["steering_range"][0],
      "max_steer": env.unwrapped.config["action"]["steering_range"][1],
    }
    planner = ARSPlanner(
      **kwargs,
    )

    mean_prev = np.zeros(2 * args.n)
    # NB! Not initializing initial velocities to goal velocity
    # led to the very first CEM samples being all over the place.
    mean_prev[:args.n] = planner.vd

  env_barrier = EnvBarrierSim(
    env.unwrapped.vehicle.WIDTH, 
    env.unwrapped.vehicle.LENGTH, 
    obs_count=env.unwrapped.config["observation"]["vehicles_count"] - 1,
    lane_count=env.unwrapped.config["lanes_count"],
    num=(args.cem_samples if args.planner == "cem" else 1),
    n=args.n,
  )

  def controls_to_action(v, steering, ego_state):
    ego_speed = np.linalg.norm(ego_state[2:4])
    v_desired = np.clip(v[1], planner.min_v, planner.max_v)
    throttle_action = (v_desired - ego_speed)/delta_t
    steering_action = steering[1]
    return np.array([throttle_action, steering_action])


  ego_state, obs = set_obs(obs, lane_ub, lane_lb)
  done = False
  delta0_prev = 0.0

  while not done:
    #print(f"ego_state: {ego_state}")
    #print(f"obs: {obs}")

    if args.planner == "cem":
      v, steering, x_traj, y_traj, theta_traj, mean_prev, x_traj_all, y_traj_all = planner.plan(
        ego_state, obs, mean_init=mean_prev, delta0=delta0_prev
      )
    elif args.planner == "nls":
      v, steering, x_traj, y_traj, theta_traj, controls_prev, costs = planner.plan(
        ego_state, obs, controls_init=controls_prev, delta0=delta0_prev
      )
    elif args.planner == "ars":
      v, steering, x_traj, y_traj, theta_traj, mean_prev, costs = planner.plan(
        ego_state, obs, mean_init=mean_prev, delta0=delta0_prev
      )

    action = controls_to_action(v, steering, ego_state)
    print(f"action: {[f'{a:.2f}' for a in action]}")
    delta0_prev = action[1]

    obs, reward, done, truncated, info = env.step(action)
    ego_state, obs = set_obs(obs, lane_ub, lane_lb)

    # Plot generated trajectories
    if args.cem_visualize_controls:
      for i in range(0, args.cem_samples, 1):
        env_barrier.lines[i].set_data(x_traj_all[i] - x_traj_all[i][0], y_traj_all[i] - y_traj_all[i][0])
    env_barrier.best_line[0].set_data(x_traj - x_traj[0], y_traj - y_traj[0])

    env_barrier.step(
      ego_state,  obs[1:], lane_lb, lane_ub
    )

    env.render()

    #input()
