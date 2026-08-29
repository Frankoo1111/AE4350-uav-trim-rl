# train.py
# --------
# Trains the TD3 agent and tracks its progress.
#
# Run from command line:
#   python train.py               # train with default settings
#   python train.py --seed 1      # different random seed
#   python train.py --steps 80000 # train longer

import os
import time
import json
import argparse
import numpy as np
import torch

from aircraft import AircraftParams
from trim_env import UAVTrimEnv, EnvConfig
from td3 import TD3, TD3Config


def evaluate(agent, env, n_episodes=10):
    """
    Test the trained agent (no exploration noise) for n_episodes.
    Returns average reward and average airspeed tracking error.
    """
    rng     = np.random.default_rng(seed=99999)
    rewards = []
    dV_list = []

    for _ in range(n_episodes):
        # pick a random commanded airspeed
        V_cmd = float(rng.uniform(env.cfg.V_min, env.cfg.V_max))
        obs, _ = env.reset(options={"V_cmd": V_cmd})

        episode_reward = 0.0
        tail_dV = []   # airspeed errors in the last 8 seconds

        for step in range(env.max_steps):
            action              = agent.act(obs, noise=0.0)   # no noise when evaluating
            obs, r, term, trunc, info = env.step(action)
            episode_reward     += r

            # collect the last 8 seconds to measure steady-state performance
            if step >= env.max_steps - int(8.0 / env.cfg.dt):
                tail_dV.append(abs(info["V"] - info["V_cmd"]))

            if term:
                break

        rewards.append(episode_reward)
        if tail_dV:
            dV_list.append(np.mean(tail_dV))

    return {
        "return_mean":    float(np.mean(rewards)),
        "return_std":     float(np.std(rewards)),
        "airspeed_error": float(np.mean(dV_list)) if dV_list else float("nan"),
    }


def train(total_steps=50000, eval_every=3000, n_eval=10,
          env_cfg=None, td3_cfg=None, verbose=True):
    """
    Main training loop.
    Returns the trained agent and its learning history.
    """
    if env_cfg is None:
        env_cfg = EnvConfig()
    if td3_cfg is None:
        td3_cfg = TD3Config()

    p = AircraftParams()

    # two separate environments: one for training, one for evaluation
    train_env = UAVTrimEnv(p, env_cfg)
    eval_env  = UAVTrimEnv(p, env_cfg)

    obs_dim = train_env.observation_space.shape[0]   # 7
    act_dim = train_env.action_space.shape[0]        # 2

    agent = TD3(obs_dim, act_dim, td3_cfg)

    # reset to obtain first observation
    train_env.action_space.seed(td3_cfg.seed)
    obs, _ = train_env.reset(seed=td3_cfg.seed)

    # record learning progress over time
    history = {
        "steps":          [],
        "eval_return":    [],
        "eval_return_std":[],
        "airspeed_error": [],
    }

    start_time = time.time()

    for step in range(1, total_steps + 1):

        # for the first start_steps: take random actions to fill the replay buffer
        # after that: use the trained policy with exploration noise
        if step < td3_cfg.start_steps:
            action = train_env.action_space.sample()
        else:
            action = agent.act(obs, noise=td3_cfg.expl_noise)

        # step the environment
        next_obs, reward, terminated, truncated, info = train_env.step(action)

        # tell the environment how far through training we are (for curriculum)
        train_env._global_step += 1

        # store experience in replay buffer
        # note: only use 'terminated' for done flag, not 'truncated'
        # (truncated just means time ran out, not that the agent failed)
        agent.buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs

        # if episode ended, start a new one
        if terminated or truncated:
            obs, _ = train_env.reset(seed=td3_cfg.seed + step)

        # learn from the replay buffer (only once we have enough data)
        if agent.buffer.size >= td3_cfg.batch_size and step >= td3_cfg.start_steps:
            agent.update()

        # periodically evaluate and print progress
        if step % eval_every == 0 or step == total_steps:
            ev = evaluate(agent, eval_env, n_episodes=n_eval)

            history["steps"].append(step)
            history["eval_return"].append(ev["return_mean"])
            history["eval_return_std"].append(ev["return_std"])
            history["airspeed_error"].append(ev["airspeed_error"])

            if verbose:
                elapsed = time.time() - start_time
                print(f"  step {step:6d}  |  "
                      f"reward {ev['return_mean']:7.1f} ± {ev['return_std']:5.1f}  |  "
                      f"airspeed error {ev['airspeed_error']:.2f} m/s  |  "
                      f"{elapsed:.0f}s elapsed")

    return {"agent": agent, "history": history}


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train TD3 to trim a fixed-wing UAV")
    ap.add_argument("--steps",   type=int,   default=50000)
    ap.add_argument("--seed",    type=int,   default=0)
    ap.add_argument("--outdir",  type=str,   default="runs/default")
    ap.add_argument("--gamma",   type=float, default=0.99)
    ap.add_argument("--lr",      type=float, default=3e-4)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cfg        = TD3Config()
    cfg.seed   = args.seed
    cfg.gamma  = args.gamma
    cfg.actor_lr  = args.lr
    cfg.critic_lr = args.lr

    print(f"Training for {args.steps} steps, seed={args.seed}")
    print(f"Saving to {args.outdir}/")
    print()

    result = train(total_steps=args.steps, td3_cfg=cfg)

    # save the trained model
    result["agent"].save(os.path.join(args.outdir, "model.pt"))

    # save the learning curve
    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump(result["history"], f, indent=2)

    print(f"\nDone! Model saved to {args.outdir}/model.pt")
