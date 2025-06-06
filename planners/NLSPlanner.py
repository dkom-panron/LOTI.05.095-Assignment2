import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random, vmap, grad, jacfwd, jacrev
import jax

class NLSPlanner:
  def __init__(self,
              n=20, 
              num_iter=100, 
              delta_t=0.1, l=2.5,
              yd=0.0, vd=10.0, min_v=0.0, max_v=15.0, 
              min_steer=-0.5, max_steer=0.5):
    self.n = n
    self.num_iter = num_iter
    self.delta_t = delta_t
    self.yd = yd
    self.vd = vd
    self.min_v = min_v
    self.max_v = max_v
    self.min_steer = min_steer
    self.max_steer = max_steer
    self.l = l

    self.w_centerline = 0.2
    self.w_centerline_last = 100.0
    self.w_smoothness = 10.0
    self.w_speed = 1.0
    self.w_lane = 0.01
    self.beta = 2.0
    self.eta = 0.1

    self.jac_func = jit(jacfwd(self.compute_error, argnums=(0)))
    #self.compute_error_func = jit(self.compute_error, static_argnums=(0,))

  @partial(jit, static_argnums=(0,))
  def clip_controls(self, controls):
    controls = controls.at[:self.n].set(jnp.clip(controls[:self.n], self.min_v, self.max_v))
    controls = controls.at[self.n:].set(jnp.clip(controls[self.n:], self.min_steer, self.max_steer))
    return controls

  """
  @partial(jit, static_argnums=(0,))
  def compute_rollout(self, controls, ego_state, delta0):
    x0, y0, vx0, vy0, theta0 = ego_state

    controls = self.clip_controls(controls)
    v = controls[:self.n]
    steering = controls[self.n:2 * self.n]

    # v0 and initial steering at 0
    v0 = jnp.sqrt(vx0**2 + vy0**2)
    #v = v.at[0].set(v0)
    #steering = steering.at[0].set(delta0)

    x_traj = jnp.zeros(self.n)
    y_traj = jnp.zeros(self.n)
    theta_traj = jnp.zeros(self.n)

    x_traj = x_traj.at[0].set(x0)
    y_traj = y_traj.at[0].set(y0)
    #theta_traj = theta_traj.at[0].set(theta0)

    # Changes in theta, x, y
    theta_changes = (v[:-1] / self.l) * jnp.tan(steering[:-1]) * self.delta_t
    theta_traj = theta_traj.at[1:].set(theta0 + jnp.cumsum(theta_changes))

    dx = v[:-1] * jnp.cos(theta_traj[:-1]) * self.delta_t
    dy = v[:-1] * jnp.sin(theta_traj[:-1]) * self.delta_t

    x_traj = x_traj.at[1:].set(x0 + jnp.cumsum(dx))
    y_traj = y_traj.at[1:].set(y0 + jnp.cumsum(dy))

    return x_traj, y_traj, theta_traj, v, steering
  """
  @partial(jit, static_argnums=(0,))
  def compute_rollout(self, controls, ego_state, delta0):
    x0, y0, vx0, vy0, theta0 = ego_state

    #controls = self.clip_controls(controls)
    v = controls[0:self.n]
    steering = controls[self.n:2*self.n]

    # Combine vx0 and vy0 into a single velocity.
    # We can calculate v0 based on what `KinematicObservation` provides us,
    # but we have to store the previous step's steering input (delta0).
    v0 = jnp.sqrt(vx0**2 + vy0**2)
    #v[0] = v0
    #steering[0] = delta0
    v = v.at[0].set(v0)
    steering = steering.at[0].set(delta0)

    x_traj = jnp.zeros(self.n)
    y_traj = jnp.zeros(self.n)
    theta_traj = jnp.zeros(self.n)

    #x_traj[0] = x0
    x_traj = x_traj.at[0].set(x0)
    #y_traj[0] = y0
    y_traj = y_traj.at[0].set(y0)
    #theta_traj[0] = theta0
    theta_traj = theta_traj.at[0].set(theta0)
    for k in range(self.n-1):
      #x_traj[k+1] = x_traj[k] + v[k] * jnp.cos(theta_traj[k]) * self.delta_t
      x_traj = x_traj.at[k+1].set(x_traj[k] + v[k] * jnp.cos(theta_traj[k]) * self.delta_t)
      #y_traj[k+1] = y_traj[k] + v[k] * jnp.sin(theta_traj[k]) * self.delta_t
      y_traj = y_traj.at[k+1].set(y_traj[k] + v[k] * jnp.sin(theta_traj[k]) * self.delta_t)
      #theta_traj[k+1] = theta_traj[k] + v[k] / self.l * jnp.tan(steering[k]) * self.delta_t
      theta_traj = theta_traj.at[k+1].set(theta_traj[k] + v[k] / self.l * jnp.tan(steering[k]) * self.delta_t)

    return x_traj, y_traj, theta_traj, v, steering

  @partial(jit, static_argnums=(0,))
  def compute_error(self, controls, ego_state, obs, delta0):
    x_traj, y_traj, theta_traj, v, steering = self.compute_rollout(controls, ego_state, delta0)

    #cost_centerline = jnp.sum((y_traj - self.yd)**2)
    error_centerline = y_traj - self.yd
    error_centerline_last = y_traj[-1] - self.yd

    #cost_smoothness = jnp.sum(jnp.diff(v)**2 + jnp.diff(steering)**2)
    error_smoothness_v = jnp.diff(v)
    error_smoothness_steering = jnp.diff(steering)

    #cost_speed = jnp.sum((v - self.vd)**2)
    error_speed = v - self.vd

    y_ub, y_lb = obs[0][0], obs[0][1]
    f_ub = y_traj - y_ub
    f_lb = -y_traj + y_lb

    #cost_lane = jnp.sum(1.0/self.beta * jnp.log(1.0 + jnp.exp(self.beta * f_lb))) \
    #          + jnp.sum(1.0/self.beta * jnp.log(1.0 + jnp.exp(self.beta * f_ub)))
    error_lane_ub = 1.0/self.beta * jnp.log(1.0 + jnp.exp(self.beta * f_lb))
    error_lane_lb = 1.0/self.beta * jnp.log(1.0 + jnp.exp(self.beta * f_ub))
    
    #jax.debug.print("error_centerline={x}", x=error_centerline)
    #jax.debug.print("error_smoothness_v={x}", x=error_smoothness_v)
    #jax.debug.print("error_smoothness_steering={x}", x=error_smoothness_steering)
    #jax.debug.print("error_speed={x}", x=error_speed)
    #jax.debug.print("error_lane_ub={x}", x=error_lane_ub)
    #jax.debug.print("error_lane_lb={x}", x=error_lane_lb)

    return jnp.hstack((
      self.w_centerline * error_centerline,
      self.w_centerline_last * error_centerline_last,
      self.w_smoothness * error_smoothness_v,
      self.w_smoothness * error_smoothness_steering,
      self.w_speed * error_speed,
      self.w_lane * error_lane_ub,
      self.w_lane * error_lane_lb
    ))

  @partial(jit, static_argnums=(0,))
  def plan(self, ego_state, obs, controls_init, delta0):
    def lax_gauss_newton(carry, _):
      X_K, state = carry

      A = self.jac_func(X_K, *state)
      b = jnp.dot(A, X_K) - self.compute_error(X_K, *state)

      Q = jnp.dot(A.T, A) + (1/self.eta) * jnp.identity(2*self.n)
      q = -jnp.dot(A.T, b) - (1/self.eta) * X_K

      X_K_new = jnp.linalg.solve(Q, -q)
      
      error_value = self.compute_error(X_K_new, *state)
      cost_current = jnp.linalg.norm(error_value)

      return (X_K_new, state), cost_current
    
    X_init = controls_init
    
    # ego_state, obs, delta0
    state = (ego_state, obs, delta0)
    
    carry_init = (X_init, state)
    carry_final, costs = jax.lax.scan(lax_gauss_newton, carry_init, jnp.arange(self.num_iter))
    
    controls_optimal = carry_final[0]

    x_traj, y_traj, theta_traj, v, steering = self.compute_rollout(controls_optimal, ego_state, delta0)
    
    return v, steering, x_traj, y_traj, theta_traj, controls_optimal, costs