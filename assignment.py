import sys
import numpy as np
import highway_env
import gymnasium as gym
from env_config import env_config

import time
import matplotlib.pyplot as plt

from assignment_args import parse_args

from highway import EnvBarrierSim, set_obs

from planners.CEMPlanner import CEMPlanner
from planners.JAXCEMPlanner import JAXCEMPlanner


if __name__ == "__main__":
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
    jax_cem_planner = JAXCEMPlanner(
      **kwargs,
    )

    #cem_planner = CEMPlanner(
    #  **kwargs,
    #)

    mean_prev = np.zeros(2 * args.n)
    # NB! Not initializing initial velocities to goal velocity
    # led to the very first CEM samples being all over the place.
    mean_prev[:args.n] = jax_cem_planner.vd
  else:
    print("Planner not implemented yet")
    sys.exit(1)

  env_barrier = EnvBarrierSim(
    env.unwrapped.vehicle.WIDTH, 
    env.unwrapped.vehicle.LENGTH, 
    obs_count=env.unwrapped.config["observation"]["vehicles_count"] - 1,
    lane_count=env.unwrapped.config["lanes_count"],
    n=args.n,
  )


  ego_state, obs = set_obs(obs, lane_ub, lane_lb)
  done = False
  delta0_prev = 0.0

  if args.cem_visualize_controls:
    fig, ax_controls = plt.subplots()
    ax_controls.set_title("Control Inputs Over Time")
    ax_controls.set_xlabel("Control Index")
    ax_controls.set_ylabel("Control Value")

  while not done:
    #print(f"ego_state: {ego_state}")
    #print(f"obs: {obs}")

    action, v, steering, x_traj, y_traj, theta_traj, mean, controls = jax_cem_planner.plan(
      ego_state, obs, mean_init=mean_prev, delta0=delta0_prev
    )

    if args.cem_visualize_controls:
      # Update control plot
      plt.cla()
      ax_controls.invert_yaxis()
      # Plot controls from JAXCEMPlanner
      for c in controls:
        _x_traj, _y_traj, _, _, _ = jax_cem_planner.compute_rollout(c, ego_state, delta0_prev)
        ax_controls.plot(_x_traj - _x_traj[0], _y_traj - _y_traj[0], "b-", alpha=0.1)
      ax_controls.plot(x_traj - x_traj[0], y_traj - y_traj[0], "r-", label="Best")

      ax_controls.grid(True)
      ax_controls.legend()
      plt.pause(0.001)

    mean_prev = mean
    delta0_prev = action[1]

    obs, reward, done, truncated, info = env.step(action)
    ego_state, obs = set_obs(obs, lane_ub, lane_lb)

    # Plot your generated trajectories here
    env_barrier.lines[0].set_data(x_traj - x_traj[0], y_traj - y_traj[0])

    env_barrier.step(
      ego_state,  obs[1:], lane_lb, lane_ub
    )

    env.render()

    #input()
