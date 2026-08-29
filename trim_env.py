# trim_env.py
# -----------
# The training environment. This is where the agent learns.
#
# Each "episode" works like this:
#   1. Pick a random target airspeed (V_cmd) between 40 and 65 m/s
#   2. Start the aircraft near that airspeed with a random attitude upset
#   3. Run for 40 seconds (800 steps of 0.05 s each)
#   4. At each step:
#        - Agent picks elevator and throttle
#        - Environment computes the resulting accelerations
#        - Reward = 1 - cost, where cost is based on those accelerations
#        - Physics are simulated forward
#
# At trim, all accelerations are zero, so reward = 1 (the maximum).

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from aircraft import AircraftParams, derivatives, rk4_step


# All the settings for the environment.
# These can be adjusted without touching the environment code.
class EnvConfig:
    def __init__(self):
        self.dt           = 0.05    # time between agent decisions [s]
        self.substeps     = 5       # physics steps per agent step (for accuracy)
        self.episode_time = 40.0    # how long each episode runs [s]
        self.V_min        = 40.0    # minimum airspeed in the envelope [m/s]
        self.V_max        = 65.0    # maximum airspeed in the envelope [m/s]
        self.V_init_band  = 2.0     # max airspeed offset at start of episode [m/s]

        # curriculum: start easy (small offset) and grow harder over training
        self.curriculum_start = 0.3       # starting offset [m/s]
        self.curriculum_steps = 30000     # steps to reach full difficulty

        # scales for the observation (keep all numbers roughly between -1 and 1)
        self.V_scale     = 12.5    # half the envelope width [m/s]
        self.accV_scale  = 0.5     # typical V_dot at a trim deviation [m/s^2]
        self.accN_scale  = 4.0     # typical normal acceleration [m/s^2]
        self.accq_scale  = 2.0     # typical pitch acceleration [rad/s^2]
        self.gamma_scale = 0.08    # typical flight-path angle [rad]
        self.qrate_scale = 0.30    # typical pitch rate [rad/s]

        # reward weights
        self.w_accel     = 1.0    # how much accelerations matter in the reward
        self.w_airspeed  = 6.0    # how much hitting the right airspeed matters
        self.w_gamma     = 1.0    # how much level flight matters
        self.w_pitchrate = 0.1    # small penalty for pitch rate
        self.w_ctrl_rate = 0.01   # small penalty for jerky control inputs
        self.cost_cap    = 12.0   # max cost per step (prevents crazy TD targets)
        self.alive_bonus = 1.0    # reward at perfect trim
        self.term_penalty = 20.0  # extra penalty if aircraft crashes out of envelope


class UAVTrimEnv(gym.Env):
    """
    The Gymnasium environment for trim discovery.
    The agent must find elevator + throttle that trims the aircraft
    at a commanded airspeed.
    """

    def __init__(self, aircraft_params=None, config=None):
        super().__init__()

        self.p   = aircraft_params if aircraft_params is not None else AircraftParams()
        self.cfg = config          if config          is not None else EnvConfig()

        # how many steps per episode.
        self.max_steps = int(round(self.cfg.episode_time / self.cfg.dt))

        # middle of the airspeed envelope (used to normalize V_cmd in observation)
        self.V_mid = 0.5 * (self.cfg.V_min + self.cfg.V_max)

        # action space: two numbers between -1 and 1.
        #   action[0] -> elevator
        #   action[1] -> throttle
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # observation space: 7 numbers (unbounded, we normalize them ourselves)
        self.observation_space = spaces.Box(
            low  = -np.inf * np.ones(7, dtype=np.float32),
            high =  np.inf * np.ones(7, dtype=np.float32),
            dtype = np.float32
        )

        # internal state: set properly in reset()
        self.state            = None
        self.V_cmd            = None
        self.prev_control     = None
        self.step_count       = 0
        self._global_step     = 0   # training loop increments this for the curriculum

    # -------------------------------------------------------------------------
    def _action_to_control(self, action):
        """Convert normalised action [-1,1] to physical elevator [rad] and throttle [0,1]."""
        action = np.clip(action, -1.0, 1.0)
        delta_e = float(action[0]) * self.p.de_limit      # scale to ±25 deg
        delta_t = float((action[1] + 1.0) * 0.5)          # map [-1,1] to [0,1]
        return np.array([delta_e, delta_t])

    def _get_observation(self, state, control):
        """
        Build the 7-number observation vector.
        All elements are normalised to be roughly between -1 and 1.
        """
        V, gamma, q, theta = state
        alpha = theta - gamma

        # compute current accelerations (the trim signal)
        xdot = derivatives(state, control, self.p)
        V_dot, gamma_dot, q_dot, _ = xdot

        obs = np.array([
            (V - self.V_cmd) / self.cfg.V_scale,          # airspeed error
            alpha,                                         # angle of attack
            q,                                             # pitch rate
            gamma,                                         # flight-path angle
            (self.V_cmd - self.V_mid) / self.cfg.V_scale, # which speed we want
            V_dot / self.cfg.accV_scale,                   # along-track acceleration
            q_dot / self.cfg.accq_scale,                   # pitch acceleration
        ], dtype=np.float32)

        return obs, xdot

    # -------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Start a new episode."""
        super().reset(seed=seed)
        rng = self.np_random

        # pick commanded airspeed for this episode
        if options and "V_cmd" in options:
            self.V_cmd = float(options["V_cmd"])
        else:
            self.V_cmd = float(rng.uniform(self.cfg.V_min, self.cfg.V_max))

        # curriculum: grow the starting offset as training progresses
        progress = min(self._global_step / self.cfg.curriculum_steps, 1.0)
        band = self.cfg.curriculum_start + progress * (self.cfg.V_init_band - self.cfg.curriculum_start)

        # randomize starting state (airspeed near V_cmd, random attitude upset)
        V0     = float(np.clip(self.V_cmd + rng.uniform(-band, band),
                               self.cfg.V_min - 2.0, self.cfg.V_max + 2.0))
        gamma0 = float(rng.uniform(-0.10, 0.10))
        q0     = float(rng.uniform(-0.10, 0.10))
        theta0 = gamma0 + float(rng.uniform(-0.05, 0.15))

        self.state        = np.array([V0, gamma0, q0, theta0])
        self.prev_control = np.array([0.0, 0.3])
        self.step_count   = 0

        obs, _ = self._get_observation(self.state, self.prev_control)
        return obs, {}

    # -------------------------------------------------------------------------
    def step(self, action):
        """
        Apply action, compute reward, step physics forward.
        Returns: observation, reward, terminated, truncated, info
        """
        control = self._action_to_control(action)

        # --- compute reward from the resulting accelerations BEFORE stepping physics ---
        V, gamma, q, theta = self.state
        xdot = derivatives(self.state, control, self.p)
        V_dot, gamma_dot, q_dot, _ = xdot
        normal_accel = V * gamma_dot   # acceleration perpendicular to velocity

        # how far off from commanded airspeed
        airspeed_error = V - self.V_cmd

        # cost from accelerations (all zero at trim)
        cost_accel = (
            (V_dot       / self.cfg.accV_scale)**2 +
            (normal_accel / self.cfg.accN_scale)**2 +
            (q_dot        / self.cfg.accq_scale)**2
        )

        # cost from not being at the right airspeed
        cost_airspeed = (airspeed_error / self.cfg.V_scale)**2

        # cost from not flying level
        cost_gamma = (gamma / self.cfg.gamma_scale)**2

        # small penalty for pitch rate
        cost_pitchrate = (q / self.cfg.qrate_scale)**2

        # small penalty for jerky control (change in control input)
        ctrl_change    = control - self.prev_control
        cost_ctrl_rate = (ctrl_change[0] / self.p.de_limit)**2 + ctrl_change[1]**2

        # total cost (capped so one bad step can't explode the training)
        cost = (
            self.cfg.w_accel     * cost_accel     +
            self.cfg.w_airspeed  * cost_airspeed  +
            self.cfg.w_gamma     * cost_gamma     +
            self.cfg.w_pitchrate * cost_pitchrate +
            self.cfg.w_ctrl_rate * cost_ctrl_rate
        )
        cost   = min(cost, self.cfg.cost_cap)
        reward = self.cfg.alive_bonus - cost

        # --- step the physics forward ---
        sub_dt = self.cfg.dt / self.cfg.substeps
        state  = self.state
        for _ in range(self.cfg.substeps):
            state = rk4_step(state, control, sub_dt, self.p)
        self.state        = state
        self.prev_control = control
        self.step_count  += 1

        # --- check if aircraft has left the safe flight envelope ---
        V_new, gamma_new, q_new, theta_new = self.state
        alpha_new  = theta_new - gamma_new
        terminated = bool(
            abs(alpha_new) > self.p.alpha_stall  or   # stall
            V_new < 28.0                          or   # too slow
            V_new > 90.0                          or   # too fast
            abs(theta_new) > np.deg2rad(60.0)         # extreme pitch
        )
        if terminated:
            reward -= self.cfg.term_penalty   # extra penalty for leaving the envelope

        truncated = bool(self.step_count >= self.max_steps)

        # build observation for next step
        obs, _ = self._get_observation(self.state, control)

        # extra info for logging (not used by the agent)
        info = {
            "V":            V_new,
            "alpha":        alpha_new,
            "gamma":        gamma_new,
            "q":            q_new,
            "V_dot":        V_dot,
            "normal_accel": normal_accel,
            "q_dot":        q_dot,
            "delta_e":      control[0],
            "delta_t":      control[1],
            "V_cmd":        self.V_cmd,
        }

        return obs, float(reward), terminated, truncated, info
