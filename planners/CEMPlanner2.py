import numpy as np

class CEMPlanner:
  def __init__(self, n=20, 
         num_samples=64, percentage_elite=0.1, num_iter=50, 
         delta_t=0.1, l=2.5,
         yd=0.0, vd=10.0, min_v=0.0, max_v=15.0, 
         min_steer=-0.5, max_steer=0.5, 
         beta=10.0):
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
    self.w_lane = 0.1

    # for warmstarting
    self.mean_prev = None

  def rollout(self, controls, ego_state):
    """
    x, y, vx, vy, theta = ego_state
    traj = np.zeros((self.n + 1, 3))
    traj[0, :] = [x, y, theta]
    for k in range(self.n):
      v = actions[k, 0]
      delta = actions[k, 1]
      x = x + v * np.cos(theta) * dt
      y = y + v * np.sin(theta) * dt
      theta = theta + (v / l) * np.tan(delta) * dt
      traj[k + 1, :] = [x, y, theta]
    return traj
    """
    x, y, vx, vy, theta = ego_state
    # initialize trajectory array with (n+1) timesteps and 3 states (x, y, theta)
    traj = np.zeros((self.n + 1, 3))
    traj[0, :] = [x, y, theta]
    
    # velocity and steering angle from actions
    v = controls[:, 0]
    delta = controls[:, 1]
    
    theta_changes = (v / self.l) * np.tan(delta) * self.delta_t
    theta_traj = theta + np.cumsum(theta_changes)
    
    dx = v * np.cos(theta_traj) * self.delta_t
    dy = v * np.sin(theta_traj) * self.delta_t
    
    x_traj = x + np.cumsum(dx)
    y_traj = y + np.cumsum(dy)
    
    traj[1:, 0] = x_traj
    traj[1:, 1] = y_traj
    traj[1:, 2] = theta_traj
    
    return traj

  def compute_cost(self, controls, ego_state, obs):
    ylb, yub = obs[0][1], obs[0][0]

    traj = self.rollout(controls, ego_state)

    y = traj[:, 1]
    v = controls[:, 0]
    delta = controls[:, 1]

    c_centerline = np.sum((y - self.yd) ** 2)
    c_smoothness = np.sum(np.diff(v) ** 2) + np.sum(np.diff(delta) ** 2)
    c_speed = np.sum((v - self.vd) ** 2)
    f_ub = y - yub
    f_lb = -y + ylb
    c_lane = np.sum(1.0 / self.beta * np.log(1 + np.exp(self.beta * f_ub))) + \
         np.sum(1.0 / self.beta * np.log(1 + np.exp(self.beta * f_lb)))

    #print(f"{c_centerline:.2f}, {c_smoothness:.2f}, {c_speed:.2f}, {c_lane:.2f}")

    return self.w_centerline * c_centerline + \
         self.w_smoothness * c_smoothness + \
         self.w_speed * c_speed + \
         self.w_lane * c_lane
    #return c_centerline

  def plan(self, ego_state, obs, mean=None):
    #ylb, yub = obs[0][1], obs[0][0]
    # initial distribution (if no warmstart, center velocity on vd and steer at zero)
    if mean is None:
      mean = np.zeros((self.n, 2))
      mean[:, 0] = self.vd # [v, steering]

    cov = np.diag([
      (self.max_v - self.min_v) * 0.2
    ] * self.n + [
      (self.max_steer - self.min_steer) * 0.2
    ] * self.n).reshape((self.n * 2, self.n * 2))

    mean = mean.reshape((-1,)) # flatten
    
    for _ in range(self.num_iter):
      # (num_samples, plan_horizon*2)
      samples = np.random.multivariate_normal(mean, cov, self.num_samples)
      samples = samples.reshape((self.num_samples, self.n, 2))

      # clip actions
      samples[:, :, 0] = np.clip(samples[:, :, 0], self.min_v, self.max_v)
      samples[:, :, 1] = np.clip(samples[:, :, 1], self.min_steer, self.max_steer)

      costs = np.zeros(self.num_samples)
      for i in range(self.num_samples):
        #traj = self.rollout(ego_state, samples[i], self.delta_t, self.l)
        costs[i] = self.compute_cost(samples[i], ego_state, obs)

      num_elite = int(self.percentage_elite * self.num_samples)
      elite_idx = np.argsort(costs)[:num_elite]
      elite = samples[elite_idx].reshape(num_elite, -1)
      mean = elite.mean(axis=0)
      cov = np.cov(elite, rowvar=False) + 1e-4 * np.eye(self.n * 2)

    # best action sequence and new mean (reshaped to (n,2))
    best_idx = np.argmin(costs)
    best_controls = samples[best_idx]

    # calculate action for HighwayEnv
    # since it expects [throttle, steer]
    # instead of [velocity, steer]
    ego_speed = np.linalg.norm(ego_state[2:4])
    v_desired = np.clip(best_controls[0, 0], self.min_v, self.max_v)
    throttle_action = (v_desired - ego_speed)
    action = np.array([throttle_action, best_controls[0, 1]])

    return best_controls, mean.reshape((self.n, 2)), action
