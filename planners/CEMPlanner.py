import numpy as np
import scipy

class CEMPlanner:
  def __init__(self, n=20, 
         num_samples=200,
         percentage_elite=0.05,
         num_iter=50, 
         delta_t=0.1,
         l=2.5,
         yd=0.0, vd=10.0,
         min_v=0.0, max_v=15.0, 
         min_steer=-0.5, max_steer=0.5, 
         beta=5.0):
    self.n = n
    self.num_samples = num_samples
    self.percentage_elite = percentage_elite
    self.num_iter = num_iter
    self.delta_t = delta_t
    self.yd = yd
    self.vd = vd
    self.min_v = min_v
    self.max_v = max_v
    self.min_steer = min_steer
    self.max_steer = max_steer
    self.beta = beta
    self.l = l
    
    self.w_centerline = 1.0
    self.w_smoothness = 1.0
    self.w_speed = 1.0
    self.w_lane = 1.0

    # for warmstarting
    init_cov_v = 2*np.identity(self.n)
    init_cov_steering = 0.1*np.identity(self.n)
    self.init_cov = scipy.linalg.block_diag(init_cov_v, init_cov_steering)

    np.random.seed(0)

  def compute_rollout(self, controls, ego_state, delta0):
    x0, y0, vx0, vy0, theta0 = ego_state

    v = controls[0:self.n].copy()
    steering = controls[self.n:2*self.n].copy()

    v0 = np.sqrt(vx0**2 + vy0**2)
    v[0] = v0
    steering[0] = delta0

    x_traj = np.zeros(self.n)
    y_traj = np.zeros(self.n)
    theta_traj = np.zeros(self.n)

    x_traj[0] = x0
    y_traj[0] = y0
    theta_traj[0] = theta0

    # Calculate changes in theta
    theta_changes = (v[:-1] / self.l) * np.tan(steering[:-1]) * self.delta_t
    theta_traj[1:] = theta0 + np.cumsum(theta_changes)

    # Calculate changes in x and y
    dx = v[:-1] * np.cos(theta_traj[:-1]) * self.delta_t
    dy = v[:-1] * np.sin(theta_traj[:-1]) * self.delta_t

    x_traj[1:] = x0 + np.cumsum(dx)
    y_traj[1:] = y0 + np.cumsum(dy)

    return x_traj, y_traj, theta_traj, v, steering

  def compute_cost(self, controls, ego_state, obs, delta0):
    x_traj, y_traj, theta_traj, v, steering = self.compute_rollout(controls, ego_state, delta0)

    cost_centerline = np.sum((y_traj - self.yd)**2)
    cost_smoothness = np.sum(np.diff(v)**2 + np.diff(steering)**2)
    cost_speed = np.sum((v - self.vd)**2)

    y_ub, y_lb = obs[0][0], obs[0][1]
    f_ub = y_traj - y_ub
    f_lb = -y_traj + y_lb
    #print(f"njx: y_ub: {y_ub}, y_lb: {y_lb}")
    print(f"{f_lb=}")
    cost_lane = np.sum(1.0/self.beta * np.log(1.0 + np.exp(self.beta * f_lb))) \
              + np.sum(1.0/self.beta * np.log(1.0 + np.exp(self.beta * f_ub)))

    print(f"{cost_centerline=}")
    print(f"{cost_smoothness=}")
    print(f"{cost_speed=}")
    print(f"{cost_lane=}")

    return (self.w_centerline * cost_centerline +
            self.w_smoothness * cost_smoothness +
            self.w_speed * cost_speed +
            self.w_lane * cost_lane)

  def plan(self, ego_state, obs, delta0, mean_init):
    cov = self.init_cov.copy()
    for i in range(self.num_iter):
      # Sample controls
      controls = np.random.multivariate_normal(mean_init, cov, self.num_samples)

      controls[:, 0:self.n] = np.clip(controls[:, 0:self.n], self.min_v, self.max_v)
      controls[:, self.n:2*self.n] = np.clip(controls[:, self.n:2*self.n], self.min_steer, self.max_steer)

      # Evaluate cost
      cost_samples = np.zeros(self.num_samples)
      for j in range(self.num_samples):
        cost_samples[j] = self.compute_cost(controls[j], ego_state, obs, delta0)

      # Select elite samples
      num_elite = int(self.percentage_elite * self.num_samples)
      elite_indices = np.argsort(cost_samples)[:num_elite]
      elite_controls = controls[elite_indices]

      # Update mean and covariance
      mean_init = np.mean(elite_controls, axis=0)
      cov = np.cov(elite_controls.T) + 0.001 * np.identity(2*self.n)
    
    controls = np.random.multivariate_normal(mean_init, cov, self.num_samples)
    cost_samples = np.zeros(self.num_samples)
    for j in range(self.num_samples):
      cost_samples[j] = self.compute_cost(controls[j], ego_state, obs, delta0)

    controls_best = controls[np.argmin(cost_samples)]
    x_traj, y_traj, theta_traj, v, steering = self.compute_rollout(controls_best, ego_state, delta0)

    # calculate action for HighwayEnv
    # ego state is [x, y, vx, vy, theta]
    ego_speed = np.linalg.norm(ego_state[2:4])
    v_desired = np.clip(v[1], self.min_v, self.max_v)
    throttle_action = (v_desired - ego_speed)/self.delta_t
    action = np.array([throttle_action, steering[1]])

    return action, v, steering, x_traj, y_traj, theta_traj, mean_init, controls, controls_best
      