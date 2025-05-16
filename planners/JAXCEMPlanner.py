import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random, vmap, grad, jacfwd, jacrev
import jax

class JAXCEMPlanner:
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
    self.w_lane = 1.0

    # for warmstarting
    #self.mean_prev = None

    self.key = random.PRNGKey(0)
    self.key, subkey = random.split(self.key)

    # "stomp-like" covariance initialization
    """
    A = np.diff(np.diff(np.identity(self.n), axis = 0), axis = 0)

    temp_1 = np.zeros(self.n)
    temp_2 = np.zeros(self.n)
    temp_3 = np.zeros(self.n)
    temp_4 = np.zeros(self.n)

    temp_1[0] = 1.0
    temp_2[0] = -2
    temp_2[1] = 1
    temp_3[-1] = -2
    temp_3[-2] = 1

    temp_4[-1] = 1.0

    A_mat = -np.vstack((temp_1, temp_2, A, temp_3, temp_4))

    R = np.dot(A_mat.T, A_mat)
    mu = np.zeros(self.n) # not needed
    cov = np.linalg.pinv(R)
    self.init_cov = jax.scipy.linalg.block_diag(0.001*cov, 0.0003*cov)
    """
    init_cov_v = 2*jnp.identity(self.n)
    init_cov_steering = 0.5*jnp.identity(self.n)

    self.init_cov = jax.scipy.linalg.block_diag(init_cov_v, init_cov_steering)

    self.compute_cost_batch = jit(vmap(self.compute_cost,
                                       in_axes=(0, None, None, None)))

  """
  @partial(jit, static_argnums=(0,))
  def compute_rollout(self, controls, ego_state):
    v_controls = controls[:self.n]
    delta_controls = controls[self.n:2*self.n]

    # Initialize state arrays
    xs = jnp.zeros(self.n + 1)
    ys = jnp.zeros(self.n + 1)
    headings = jnp.zeros(self.n + 1)

    #xs[0], ys[0], _, _, headings[0] = ego_state
    #xs[0] = ego_state[0]
    xs = xs.at[0].set(ego_state[0])
    ys = ys.at[0].set(ego_state[1])
    headings = headings.at[0].set(ego_state[4])
    
    for k in range(self.n):
        #xs[k+1] = xs[k] + v_controls[k] * jnp.cos(headings[k]) * self.delta_t
        xs = xs.at[k+1].set(xs[k] + v_controls[k] * jnp.cos(headings[k]) * self.delta_t)
        #ys[k+1] = ys[k] + v_controls[k] * jnp.sin(headings[k]) * self.delta_t
        ys = ys.at[k+1].set(ys[k] + v_controls[k] * jnp.sin(headings[k]) * self.delta_t)
        #headings[k+1] = headings[k] + (v_controls[k] / self.l) * jnp.tan(delta_controls[k]) * self.delta_t
        headings = headings.at[k+1].set(headings[k] + (v_controls[k] / self.l) * jnp.tan(delta_controls[k]) * self.delta_t)

    # Optionally return as a stacked array of shape (n_steps+1, 3) for (x, y, heading)
    #rollout = np.stack([xs, ys, headings], axis=-1)
    return xs, ys, v_controls, delta_controls
  """
  @partial(jit, static_argnums=(0,))
  def compute_rollout(self, controls, ego_state, delta0):
    x0, y0, vx0, vy0, theta0 = ego_state

    v = controls[:self.n]
    steering = controls[self.n:2 * self.n]

    # v0 and initial steering at 0
    v0 = jnp.sqrt(vx0**2 + vy0**2)
    v = v.at[0].set(v0)
    steering = steering.at[0].set(delta0)

    x_traj = jnp.zeros(self.n)
    y_traj = jnp.zeros(self.n)
    theta_traj = jnp.zeros(self.n)

    x_traj = x_traj.at[0].set(x0)
    y_traj = y_traj.at[0].set(y0)
    theta_traj = theta_traj.at[0].set(theta0)

    # Changes in theta, x, y
    theta_changes = (v[:-1] / self.l) * jnp.tan(steering[:-1]) * self.delta_t
    theta_traj = theta_traj.at[1:].set(theta0 + jnp.cumsum(theta_changes))

    dx = v[:-1] * jnp.cos(theta_traj[:-1]) * self.delta_t
    dy = v[:-1] * jnp.sin(theta_traj[:-1]) * self.delta_t

    x_traj = x_traj.at[1:].set(x0 + jnp.cumsum(dx))
    y_traj = y_traj.at[1:].set(y0 + jnp.cumsum(dy))

    return x_traj, y_traj, theta_traj, v, steering

  @partial(jit, static_argnums=(0,))
  def compute_cost(self, controls, ego_state, obs, delta0):
    x_traj, y_traj, theta_traj, v, steering = self.compute_rollout(controls, ego_state, delta0)

    cost_centerline = jnp.sum((y_traj - self.yd) ** 2)
    cost_smoothness = jnp.sum(jnp.diff(v) ** 2) + jnp.sum(jnp.diff(steering) ** 2)
    cost_speed = jnp.sum((v - self.vd) ** 2)

    ylb, yub = obs[0][1], obs[0][0]
    f_ub = y_traj - yub
    f_lb = -y_traj + ylb
    cost_lane = jnp.sum(1.0 / self.beta * jnp.log(1 + jnp.exp(self.beta * f_ub))) + \
             jnp.sum(1.0 / self.beta * jnp.log(1 + jnp.exp(self.beta * f_lb)))

    return (self.w_centerline * cost_centerline +
            self.w_smoothness * cost_smoothness +
            self.w_speed * cost_speed +
            self.w_lane * cost_lane)

  @partial(jit, static_argnums=(0,))
  def plan(self, ego_state, obs, mean_init, delta0):
    def lax_cem(carry, _):
      mean, cov, key = carry
      key, subkey = random.split(key)

      controls = jax.random.multivariate_normal(subkey, mean, cov, (self.num_samples,))
      controls = controls.at[..., :self.n].set(jnp.clip(controls[..., :self.n], self.min_v, self.max_v))
      controls = controls.at[..., self.n:].set(jnp.clip(controls[..., self.n:], self.min_steer, self.max_steer))

      cost_samples = self.compute_cost_batch(
        controls,
        ego_state,
        obs,
        delta0
      )
      
      idx = jnp.argsort(cost_samples)
      elite_num = int(self.percentage_elite * self.num_samples)
      controls_elite = controls[idx[:elite_num]]
      
      new_mean = jnp.mean(controls_elite, axis=0)
      new_cov = jnp.cov(controls_elite.T) + 0.001*jnp.identity(2*self.n)
      
      return (new_mean, new_cov, key), None

    init_carry = (mean_init, self.init_cov, self.key)
    
    (final_mean, final_cov, final_key), _ = jax.lax.scan(
        lax_cem, 
        init_carry, 
        jnp.arange(self.num_iter)
    )
    
    key, subkey = random.split(final_key)
    controls = jax.random.multivariate_normal(subkey, final_mean, final_cov, (self.num_samples,))
    cost_samples = self.compute_cost_batch(
        controls,
        ego_state,
        obs,
        delta0
    )
    
    controls_best = controls[jnp.argmin(cost_samples)]
    x_traj, y_traj, theta_traj, v, steering = self.compute_rollout(controls_best, ego_state, delta0)

    ego_speed = jnp.linalg.norm(ego_state[2:4])
    v_desired = jnp.clip(v[1], self.min_v, self.max_v)
    throttle_action = (v_desired - ego_speed)/self.delta_t
    action = jnp.array([throttle_action, steering[1]])

    return action, v, steering, x_traj, y_traj, theta_traj, final_mean