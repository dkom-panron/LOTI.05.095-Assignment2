import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random, vmap, grad, jacfwd, jacrev
import jax

class ARSPlanner:
  def __init__(self,
               n=20, 
         num_samples=64, # N
         percentage_elite=0.1, # b 
         num_iter=50, # num_step
         delta_t=0.1, l=2.5,
         yd=0.0, vd=10.0, min_v=0.0, max_v=15.0, 
         min_steer=-0.5, max_steer=0.5):
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
    self.l = l

    self.w_centerline = 10.0
    self.w_smoothness = 50.0
    self.w_speed = 1.0
    self.w_lane = 1.0
    self.beta = 5.0

    self.alpha = 0.01
    self.nu = 0.1

    self.key = random.PRNGKey(0)
    self.key, subkey = random.split(self.key)

    self.compute_cost_batch = jit(vmap(self.compute_cost,
                                    in_axes=(0, None, None, None)))

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

    cost_centerline = jnp.sum((y_traj - self.yd)**2)
    cost_smoothness = jnp.sum(jnp.diff(v)**2 + jnp.diff(steering)**2)
    cost_speed = jnp.sum((v - self.vd)**2)

    y_ub, y_lb = obs[0][0], obs[0][1]
    f_ub = y_traj - y_ub
    f_lb = -y_traj + y_lb

    cost_lane = jnp.sum(1.0/self.beta * jnp.log(1.0 + jnp.exp(self.beta * f_lb))) \
              + jnp.sum(1.0/self.beta * jnp.log(1.0 + jnp.exp(self.beta * f_ub)))
    
    return (self.w_centerline * cost_centerline +
            self.w_smoothness * cost_smoothness +
            self.w_speed * cost_speed +
            self.w_lane * cost_lane), x_traj, y_traj

  @partial(jit, static_argnums=(0,))
  def clip_controls(self, controls):
    controls = controls.at[..., :self.n].set(jnp.clip(controls[..., :self.n], self.min_v, self.max_v))
    controls = controls.at[..., self.n:].set(jnp.clip(controls[..., self.n:], self.min_steer, self.max_steer))
    return controls

  @partial(jit, static_argnums=(0,))
  def plan(self, ego_state, obs, mean_init, delta0):
    def ars_step(mean, key):
      # Generate perturbations
      key, subkey = random.split(key)
      deltas = random.normal(subkey, (self.num_samples, *mean.shape))

      deltas_positive = mean[None] + self.nu * deltas
      deltas_negative = mean[None] - self.nu * deltas

      # Compute costs for positive and negative perturbations
      cost_positive, _, _ = self.compute_cost_batch(deltas_positive, ego_state, obs, delta0)
      cost_negative, _, _ = self.compute_cost_batch(deltas_negative, ego_state, obs, delta0)

      # Select top b elite samples
      max_cost = jnp.maximum(cost_positive, cost_negative)
      idx = jnp.argsort(max_cost)[:int(self.percentage_elite * self.num_samples)]
      deltas_elite = deltas[idx]
      cost_diff = cost_positive[idx] - cost_negative[idx]

      # Compute gradient
      weighted_deltas = deltas_elite * cost_diff[..., None]
      grad = jnp.sum(weighted_deltas, axis=0)

      # Update mean
      std_cost = jnp.std(jnp.hstack([cost_positive[idx], cost_negative[idx]])).clip(1e-6)
      mean = mean + (self.alpha / (std_cost * self.percentage_elite * self.num_samples)) * grad

      return mean, key

    mean = mean_init
    key = self.key

    for _ in range(self.num_iter):
      mean, key = ars_step(mean, key)

    controls = mean
    x_traj, y_traj, theta_traj, v, steering = self.compute_rollout(controls, ego_state, delta0)

    #jax.debug.print("x_traj={x}", x=x_traj)
    #jax.debug.print("y_traj={y}", y=y_traj)
    #jax.debug.print("theta_traj={theta}", theta=theta_traj)
    #jax.debug.print("v={v}", v=v)
    #jax.debug.print("steering={s}", s=steering)
    return v, steering, x_traj, y_traj, theta_traj, mean
