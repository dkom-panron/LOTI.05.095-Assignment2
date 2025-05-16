env_config = dict(
    other_vehicles_type="highway_env.vehicle.behavior.IDMVehicle",
    manual_control=False,
    lanes_count=5,
    vehicles_count=0,
    vehicles_density=2.0,
    controlled_vehicles=1,
    initial_lane_id=0,
    ego_spacing=2.0,
    duration=100,
    speed_limit=15,
    simulation_frequency=100,
    policy_frequency=10,

    obs_normalize=False,
    observation = dict(
        type="Kinematics",
        vehicles_count=2,
        features=["x", "y", "vx", "vy", "heading"],
        absolute=False,
        clip=False,
        normalize=False, 
        see_behind=False,
    ),   

    action=dict(
        type="ContinuousAction",
        clip=True,                      # Clip action to defined range
        longitudinal=True,              # Enable throttle control
        lateral=True,                   # Enable steerig control
        dynamical=True,                 # Enable dynamics in simulation (friction/slip) rather than Kinematics
        acceleration_range=[-10, 10],   # Range of acceleration values [m/s2]
        steering_range=[-0.2, 0.2],   # Range of steering values [rad]
        speed_range=[-20, 50],          # Range of reachable speeds [m/s]
    ),
    right_lane_reward=0,
    high_speed_reward=10.0,
    collision_reward=1,
    lane_change_reward=0,
    reward_speed_range=[0, 40],
    normalize_reward=True,
    on_road_reward=0.1,
    offroad_terminal=False,
    collision_terminal=True,
    screen_width=600,   # [px]
    screen_height=400,  # [px]
    centering_position=[0.3, 0.5],
    scaling=5.5,
    show_trajectories=False,
)
