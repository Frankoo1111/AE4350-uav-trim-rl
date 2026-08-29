"""
analyze.py
==========
Produce the figures for the report:

  1. Learning curve (mean +/- std over random seeds)            -> the LEARNING EFFECT
  2. Learned trim schedule vs. analytical ground truth          -> ANALYSIS OF SOLUTION
  3. Example closed-loop time-histories at two airspeeds        -> behaviour / reproducibility

Usage
-----
  python3 analyze.py --train-seeds 3 --steps 50000      # train then plot
  python3 analyze.py                                    # reuse cached runs/
"""

from __future__ import annotations
import argparse, os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from aircraft import AircraftParams, derivatives
from trim_env import UAVTrimEnv, EnvConfig
from td3 import TD3, TD3Config
from train import train
from analytical_trim import trim_schedule, trim_at

RUNDIR = "runs/main"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


# ----------------------------------------------------------------- training the helper
def train_seeds(n_seeds, steps, eval_every=3000):
    os.makedirs(RUNDIR, exist_ok=True)
    for s in range(n_seeds):
        print(f"\n=== training seed {s} ===")
        cfg = TD3Config()
        cfg.seed = s
        cfg.start_steps = 5000
        out = train(total_steps=steps, eval_every=eval_every, n_eval=10, td3_cfg=cfg)
        out["agent"].save(os.path.join(RUNDIR, f"model_seed{s}.pt"))
        with open(os.path.join(RUNDIR, f"history_seed{s}.json"), "w") as f:
            json.dump(out["history"], f)


# ----------------------------------------------------------------- rollout the helper
def rollout(agent, env, V_cmd, init_state=None, deterministic=True):
    """Roll out the deterministic policy; return arrays of the trajectory."""
    env.reset(options={"V_cmd": V_cmd})
    if init_state is not None:
        env.state = np.array(init_state, dtype=float)
        env.prev_control = np.array([0.0, 0.5])
    obs, _ = env._get_observation(env.state, env.prev_control)
    traj = {k: [] for k in ("t", "V", "alpha", "gamma", "q",
                            "delta_e", "delta_t", "V_dot", "q_dot", "n_acc")}
    for t in range(env.max_steps):
        a = agent.act(obs, noise=0.0)
        obs, r, term, trunc, info = env.step(a)
        traj["t"].append(t * env.cfg.dt)
        traj["V"].append(info["V"]); traj["alpha"].append(info["alpha"])
        traj["gamma"].append(info["gamma"]); traj["q"].append(info["q"])
        traj["delta_e"].append(info["delta_e"]); traj["delta_t"].append(info["delta_t"])
        traj["V_dot"].append(info["V_dot"]); traj["q_dot"].append(info["q_dot"])
        traj["n_acc"].append(info["normal_accel"])
        if term:
            break
    return {k: np.array(v) for k, v in traj.items()}


# ----------------------------------------------------------------- figure 1
def plot_learning_curve():
    files = sorted(glob.glob(os.path.join(RUNDIR, "history_seed*.json")))
    if not files:
        print("no histories found; skipping learning curve")
        return
    curves = [json.load(open(f)) for f in files]
    L = min(len(c["steps"]) for c in curves)   # align to shortest run
    steps = np.array(curves[0]["steps"][:L])
    R = np.array([c["eval_return"][:L] for c in curves])
    dV = np.array([c["ss_airspeed_err"][:L] for c in curves])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    m, s = R.mean(0), R.std(0)
    ax1.plot(steps, m, color="C0", lw=2, label=f"mean of {len(files)} seeds")
    ax1.fill_between(steps, m - s, m + s, color="C0", alpha=0.25, label="+/- 1 std")
    ax1.set_xlabel("environment steps"); ax1.set_ylabel("evaluation return")
    ax1.set_title("Learning curve"); ax1.legend(); ax1.grid(alpha=0.3)

    mV = np.nanmean(dV, 0)
    sV = np.nanstd(dV, 0)
    ax2.plot(steps, mV, color="C3", lw=2)
    ax2.fill_between(steps, np.maximum(mV - sV, 0), mV + sV, color="C3", alpha=0.25)
    ax2.set_xlabel("environment steps")
    ax2.set_ylabel("steady-state |V - V_cmd|  [m/s]")
    ax2.set_title("Trim airspeed error vs. training"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_learning_curve.png"), dpi=140)
    print("saved fig1_learning_curve.png")

# ----------------------------------------------------------------- figure 2
def plot_trim_schedule():
    model = os.path.join(RUNDIR, "model_seed0.pt")
    if not os.path.exists(model):
        print("no model found; skipping trim schedule")
        return
    p = AircraftParams(); cfg = EnvConfig()
    env = UAVTrimEnv(p, cfg)
    agent = TD3(env.observation_space.shape[0], env.action_space.shape[0],
                TD3Config())
    agent.load(model)

    Vgrid = np.linspace(cfg.V_min, cfg.V_max, 13)
    ana = trim_schedule(Vgrid, p)

    learned = {k: [] for k in ("delta_e", "delta_t", "alpha")}
    tail = int(2.0 / cfg.dt)
    for V in Vgrid:
        tr = rollout(agent, env, V, init_state=[V, 0.06, -0.04, 0.12])
        for k in learned:
            learned[k].append(np.mean(tr[k][-tail:]))
    learned = {k: np.array(v) for k, v in learned.items()}

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    specs = [("alpha", "angle of attack [deg]", np.rad2deg),
             ("delta_e", "elevator [deg]", np.rad2deg),
             ("delta_t", "throttle [-]", lambda x: x)]
    for ax, (key, ylab, conv) in zip(axs, specs):
        ax.plot(Vgrid, conv(ana[key]), "k--", lw=2, label="analytical trim")
        ax.plot(Vgrid, conv(learned[key]), "o-", color="C0", ms=5,
                label="learned (RL)")
        ax.set_xlabel("commanded airspeed [m/s]"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Learned trim schedule vs. analytical ground truth")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_trim_schedule.png"), dpi=140)

    # quantitative agreement
    mae_de = np.rad2deg(np.mean(np.abs(learned["delta_e"] - ana["delta_e"])))
    mae_dt = np.mean(np.abs(learned["delta_t"] - ana["delta_t"]))
    mae_al = np.rad2deg(np.mean(np.abs(learned["alpha"] - ana["alpha"])))
    print(f"saved fig2_trim_schedule.png | MAE elevator={mae_de:.2f} deg, "
          f"throttle={mae_dt:.3f}, alpha={mae_al:.2f} deg")


# ----------------------------------------------------------------- figure 3
def plot_time_histories():
    model = os.path.join(RUNDIR, "model_seed0.pt")
    if not os.path.exists(model):
        print("no model found; skipping time histories")
        return
    p = AircraftParams(); cfg = EnvConfig()
    env = UAVTrimEnv(p, cfg)
    agent = TD3(env.observation_space.shape[0], env.action_space.shape[0],
                TD3Config())
    agent.load(model)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for Vc, col in [(46.0, "C0"), (56.0, "C1")]:
        # start 3 m/s below Vc with attitude upset
        tr = rollout(agent, env, Vc,
                     init_state=[Vc, 0.06, -0.04, 0.12])
        ta = trim_at(Vc, p)
        axs[0, 0].plot(tr["t"], tr["V"], col, label=f"V_cmd={Vc:.0f}")
        axs[0, 0].axhline(Vc, color=col, ls=":", alpha=0.6)
        axs[0, 1].plot(tr["t"], np.rad2deg(tr["alpha"]), col)
        axs[0, 1].axhline(np.rad2deg(ta["alpha"]), color=col, ls=":", alpha=0.6)
        axs[1, 0].plot(tr["t"], np.rad2deg(tr["delta_e"]), col)
        axs[1, 0].axhline(np.rad2deg(ta["delta_e"]), color=col, ls=":", alpha=0.6)
        axs[1, 1].plot(tr["t"], tr["delta_t"], col)
        axs[1, 1].axhline(ta["delta_t"], color=col, ls=":", alpha=0.6)

    axs[0, 0].set_ylabel("airspeed V [m/s]"); axs[0, 0].set_title("airspeed (dotted = command)")
    axs[0, 1].set_ylabel("alpha [deg]"); axs[0, 1].set_title("angle of attack (dotted = analytical trim)")
    axs[1, 0].set_ylabel("elevator [deg]"); axs[1, 0].set_title("elevator command (dotted = analytical trim)")
    axs[1, 1].set_ylabel("throttle [-]"); axs[1, 1].set_title("throttle command (dotted = analytical trim)")
    for ax in axs.ravel():
        ax.set_xlabel("time [s]"); ax.grid(alpha=0.3)
    axs[0, 0].legend()
    fig.suptitle("Closed-loop trimming from an off-trim initial condition")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_time_histories.png"), dpi=140)
    print("saved fig3_time_histories.png")


def train_one_seed(seed, steps, eval_every=3000):
    os.makedirs(RUNDIR, exist_ok=True)
    print(f"=== training seed {seed} ===")
    cfg = TD3Config()
    cfg.seed = seed
    cfg.start_steps = 5000
    out = train(total_steps=steps, eval_every=eval_every, n_eval=10, td3_cfg=cfg)
    out["agent"].save(os.path.join(RUNDIR, f"model_seed{seed}.pt"))
    with open(os.path.join(RUNDIR, f"history_seed{seed}.json"), "w") as f:
        json.dump(out["history"], f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, default=0,
                    help="if >0, train this many seeds before plotting")
    ap.add_argument("--one-seed", type=int, default=-1,
                    help="train just this single seed, then exit")
    ap.add_argument("--steps", type=int, default=100000)
    args = ap.parse_args()
    if args.one_seed >= 0:
        train_one_seed(args.one_seed, args.steps)
        return
    if args.train_seeds > 0:
        train_seeds(args.train_seeds, args.steps)
    plot_learning_curve()
    plot_trim_schedule()
    plot_time_histories()


if __name__ == "__main__":
    main()
