# td3.py
# ------
# The TD3 learning algorithm.
#
# TD3 = Twin Delayed Deep Deterministic Policy Gradient
#
# It has three parts:
#   Actor  -- the policy (takes observation, outputs action)
#   Critic -- estimates how good a state+action pair is (x2 for stability)
#   ReplayBuffer -- stores past experiences for the agent to learn from
#
# The three tricks that make TD3 better than basic DDPG:
#   1. Two critics, always use the lower estimate  -> avoids overoptimism
#   2. Update the actor less often than critics    -> more stable
#   3. Add noise to target actions                 -> smoother value estimates

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ settings
class TD3Config:
    def __init__(self):
        self.hidden_size    = 128      # neurons in each hidden layer
        self.gamma          = 0.99     # discount factor (how much future reward matters)
        self.tau            = 0.005    # how fast target networks follow main networks
        self.actor_lr       = 3e-4     # learning rate for the actor
        self.critic_lr      = 3e-4     # learning rate for the critics
        self.batch_size     = 256      # how many experiences per learning step
        self.buffer_size    = 200000   # max experiences stored
        self.start_steps    = 5000     # random actions for this many steps before learning
        self.expl_noise     = 0.1      # exploration noise during training
        self.policy_noise   = 0.2      # noise on target actions
        self.noise_clip     = 0.5      # max size of that noise
        self.policy_delay   = 2        # update actor every N critic updates
        self.seed           = 0


# ------------------------------------------------------------------ actor
class Actor(nn.Module):
    """
    The policy network.
    Input:  7-number observation
    Output: 2-number action (elevator, throttle), both in (-1, 1)
    """
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh()    # squashes output to (-1, 1)
        )

    def forward(self, obs):
        return self.net(obs)


# ------------------------------------------------------------------ critic
class Critic(nn.Module):
    """
    Two Q-networks that estimate expected total future reward.
    Input:  observation + action concatenated
    Output: one number (the estimated value)
    """
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        input_size = obs_dim + act_dim

        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Q2 (same structure, different weights)
        self.q2 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, act):
        # used when updating the actor (only need one critic)
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x)


# ------------------------------------------------------------------ replay buffer
class ReplayBuffer:
    """
    Stores past (observation, action, reward, next_observation, done) tuples.
    Works like a circular list -- old entries get overwritten when full.
    """
    def __init__(self, obs_dim, act_dim, capacity):
        self.capacity = capacity
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act      = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew      = np.zeros((capacity, 1),       dtype=np.float32)
        self.done     = np.zeros((capacity, 1),       dtype=np.float32)
        self.ptr      = 0   # where to write next
        self.size     = 0   # how many are stored right now

    def add(self, obs, act, rew, next_obs, done):
        """Store one experience."""
        self.obs[self.ptr]      = obs
        self.act[self.ptr]      = act
        self.rew[self.ptr]      = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr]     = done
        self.ptr  = (self.ptr + 1) % self.capacity   # wrap around
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        """Pick a random batch to learn from."""
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.act[idx]),
            torch.as_tensor(self.rew[idx]),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.done[idx]),
        )


# ------------------------------------------------------------------ TD3 agent
class TD3:
    """The full TD3 agent."""

    def __init__(self, obs_dim, act_dim, cfg):
        self.cfg     = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device  = torch.device("cpu")

        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        h = cfg.hidden_size

        # main networks (trained every step)
        self.actor  = Actor(obs_dim, act_dim, h)
        self.critic = Critic(obs_dim, act_dim, h)

        # target networks (slow copies, used to compute stable TD targets)
        self.actor_target  = Actor(obs_dim, act_dim, h)
        self.critic_target = Critic(obs_dim, act_dim, h)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.buffer   = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size)
        self.update_count = 0   # counts how many times update() has been called

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, noise=0.0):
        """
        Pick an action given the current observation.
        noise = 0 for evaluation, noise > 0 during training for exploration.
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action     = self.actor(obs_tensor).numpy().flatten()

        if noise > 0.0:
            action = action + self.rng.normal(0, noise, size=self.act_dim)

        return np.clip(action, -1.0, 1.0)

    # -------------------------------------------------------------------------
    def update(self):
        """One learning step -- update critics, and occasionally the actor."""
        self.update_count += 1

        obs, act, rew, next_obs, done = self.buffer.sample(self.cfg.batch_size, self.rng)

        # ---- update the critics ----
        with torch.no_grad():
            # trick 3: add noise to target action for smoothing
            noise    = (torch.randn_like(act) * self.cfg.policy_noise
                       ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_act = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)

            # trick 1: use the LOWER of the two target critic estimates
            q1_t, q2_t = self.critic_target(next_obs, next_act)
            q_target    = rew + self.cfg.gamma * (1.0 - done) * torch.min(q1_t, q2_t)

        # train critics to predict q_target
        q1, q2      = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- update the actor (trick 2: less often than critics) ----
        if self.update_count % self.cfg.policy_delay == 0:
            # actor loss: maximise Q (so minimise -Q)
            actor_loss = -self.critic.q1_only(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # slowly blend target networks toward main networks
            self._soft_update()

    def _soft_update(self):
        """Polyak averaging: target = (1-tau)*target + tau*main"""
        tau = self.cfg.tau
        for main, target in [(self.actor, self.actor_target),
                              (self.critic, self.critic_target)]:
            for p_main, p_target in zip(main.parameters(), target.parameters()):
                p_target.data.mul_(1 - tau).add_(tau * p_main.data)

    # -------------------------------------------------------------------------
    def save(self, path):
        """Save the trained networks to a file."""
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        """Load previously trained networks from a file."""
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.actor_target.load_state_dict(data["actor"])
        self.critic_target.load_state_dict(data["critic"])
