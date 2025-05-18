import argparse

def parse_args():
  parser = argparse.ArgumentParser(description="HighwayEnv planner, made for LOTI.05.095 second assignment.")
  
  parser.add_argument(
    "--planner",
    type=str,
    choices=["cem", "nls", "ars"],
    default="cem",
    help="Planner type: 'cem', 'nls', or 'ars'"
  )

  # Envionment specific arguments
  parser.add_argument(
    "--env-initial-lane-id",
    type=int,
    default=0,
    help="Initial lane id for the ego vehicle."
  )

  # Goal related arguments
  parser.add_argument(
    "--goal-yd",
    type=float,
    default=8.0,
    help="Centerline y coordinate for the goal. Each lane is 4 units wide."
  )
  parser.add_argument(
    "--goal-vd",
    type=float,
    default=20.0,
    help="Desired speed for the goal."
  )

  # Planner parameters
  parser.add_argument(
    "--env-vehicles-count",
    type=int,
    default=2,
    help="Number of vehicles in the environment."
  )

  parser.add_argument(
    "--n",
    type=int,
    default=20,
    help="Number of control inputs (n * linear velocities + n * steering commands)."
  )

  # CEM specific arguments
  parser.add_argument(
    "--cem-samples",
    type=int,
    default=200,
    help="Number of samples for CEM."
  )
  parser.add_argument(
    "--cem-elite",
    type=float,
    default=0.05,
    help="Percentage of elite samples for CEM."
  )
  parser.add_argument(
    "--cem-iter",
    type=int,
    default=50,
    help="Number of iterations for CEM."
  )
  parser.add_argument(
    "--cem-stomp-like",
    type=bool,
    default=True,
    help="Use stomp-like covariance initialization for CEM."
  )
  parser.add_argument(
    "--cem-visualize-controls",
    type=bool,
    default=False,
    help="Visualize CEM controls."
  )
  
  return parser.parse_args()
