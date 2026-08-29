"""
demo.py  --  Before/after visualization of the TD3 trim discovery result.

Three-part figure:
  Left   -- random untrained policy (chaotic, no convergence)
  Middle -- trained TD3 policy      (pitch settles within ~2 s)
  Bottom -- learned vs analytical trim schedule across 40-65 m/s

The initial condition for both time-history panels is the same:
  aircraft at the commanded airspeed with a pitch attitude and rate upset.
This isolates the pitch-trim task; what the policy learned to do well.
Airspeed convergence is slower due to 1000 kg inertia (Vdot ~ 0.1-0.3 m/s^2)
and is reflected in both panels.

Usage:
    python3 demo.py                       # shows interactive window
    python3 demo.py --save result.png     # saves figure
    python3 demo.py --Vcmd 45.0           # different airspeed
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from aircraft import AircraftParams
from analytical_trim import trim_at, trim_schedule
from trim_env import UAVTrimEnv, EnvConfig
from td3 import TD3, TD3Config


def rollout(agent, env, Vcmd, init_state, rng=None):
    """Roll out a policy. agent=None means random."""
    env.reset(options={"V_cmd": Vcmd})
    env.state = np.array(init_state, dtype=float)
    env.prev_action_phys = np.array([0.0, 0.5])
    obs, _ = env._observe(env.state, env.prev_action_phys)
    traj = dict(t=[], V=[], alpha=[], delta_e=[], delta_t=[], Vdot=[], qdot=[])
    for step in range(env.max_steps):
        a = rng.uniform(-1, 1, 2) if agent is None else agent.act(obs, 0.0)
        obs, _, term, _, info = env.step(a)
        traj["t"].append(step * env.cfg.dt)
        traj["V"].append(info["V"])
        traj["alpha"].append(np.rad2deg(info["alpha"]))
        traj["delta_e"].append(np.rad2deg(info["delta_e"]))
        traj["delta_t"].append(info["delta_t"])
        traj["Vdot"].append(info["V_dot"])
        traj["qdot"].append(info["q_dot"])
        if term: break
    return {k: np.array(v) for k, v in traj.items()}


def envelope_sweep(agent, env, p, cfg, n=11):
    Vg  = np.linspace(cfg.V_min, cfg.V_max, n)
    ana = trim_schedule(Vg, p)
    tail = int(8.0 / cfg.dt)
    l_de, l_dt, l_al = [], [], []
    for V in Vg:
        t = trim_at(V, p)
        env.reset(options={"V_cmd": V})
        env.state = np.array([V, 0., 0., t["theta"]])
        env.prev_action_phys = np.array([t["delta_e"], t["delta_t"]])
        obs, _ = env._observe(env.state, env.prev_action_phys)
        des, dts, als = [], [], []
        for _ in range(env.max_steps):
            a = agent.act(obs, 0.0)
            obs, _, te, _, info = env.step(a)
            des.append(info["delta_e"])
            dts.append(info["delta_t"])
            als.append(info["alpha"])
            if te: break
        l_de.append(np.mean(des[-tail:]))
        l_dt.append(np.mean(dts[-tail:]))
        l_al.append(np.mean(als[-tail:]))
    return Vg, ana, np.array(l_de), np.array(l_dt), np.array(l_al)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/main/model_seed0.pt")
    ap.add_argument("--Vcmd",  type=float, default=52.5)
    ap.add_argument("--save",  default="")
    args = ap.parse_args()

    p   = AircraftParams()
    cfg = EnvConfig()
    env = UAVTrimEnv(p, cfg)
    rng = np.random.default_rng(42)

    agent = TD3(env.observation_space.shape[0], env.action_space.shape[0], TD3Config())
    agent.load(args.model)

    Vcmd  = args.Vcmd
    ta    = trim_at(Vcmd, p)
    # start at commanded airspeed, attitude + pitchrate upset only
    init = [Vcmd, 0.06, -0.04, 0.12]

    print(f"Random policy rollout   (Vcmd={Vcmd}) ...")
    rand_traj = rollout(None,  env, Vcmd, init, rng=rng)
    print(f"Trained policy rollout  (Vcmd={Vcmd}) ...")
    td3_traj  = rollout(agent, env, Vcmd, init)
    print("Envelope sweep ...")
    Vg, ana, l_de, l_dt, l_al = envelope_sweep(agent, env, p, cfg)

    mae_de = np.rad2deg(np.mean(np.abs(l_de - ana["delta_e"])))
    mae_dt = np.mean(np.abs(l_dt - ana["delta_t"]))
    mae_al = np.rad2deg(np.mean(np.abs(l_al - ana["alpha"])))

    # ---------------------------------------------------------------- figure
    CR = "#d94f4f"   # red  = untrained
    CB = "#1a73e8"   # blue = trained
    CA = "#222222"   # dark = analytical

    fig, axes = plt.subplots(4, 3, figsize=(15, 12),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.38})
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.98,
             "Trim Discovery: Untrained vs Trained TD3 Agent  —  1000 kg MALE UAV",
             ha="center", va="top", fontsize=14, fontweight="bold")

    # ---- time history panels (rows 0-3, cols 0 and 1) ----
    rows = [
        ("V",       "Airspeed [m/s]",      Vcmd,
         f"$V_{{cmd}}={Vcmd}$ m/s"),
        ("alpha",   r"AoA $\alpha$ [deg]", np.rad2deg(ta["alpha"]),
         fr"trim $\alpha^*={np.rad2deg(ta['alpha']):.2f}°$"),
        ("delta_e", "Elevator [deg]",       np.rad2deg(ta["delta_e"]),
         fr"trim $\delta_e^*={np.rad2deg(ta['delta_e']):.2f}°$"),
        ("delta_t", "Throttle [–]",         ta["delta_t"],
         fr"trim $\delta_t^*={ta['delta_t']:.3f}$"),
    ]

    for ri, (key, ylabel, ref, reflabel) in enumerate(rows):
        for ci, (traj, col, title) in enumerate([
            (rand_traj, CR, f"BEFORE — random policy  (Vcmd={Vcmd} m/s)"),
            (td3_traj,  CB, f"AFTER  — trained TD3    (Vcmd={Vcmd} m/s)"),
        ]):
            ax = axes[ri, ci]
            ax.plot(traj["t"], traj[key], color=col, lw=2.0)
            ax.axhline(ref, color=CA, lw=1.4, ls="--", label=reflabel)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.22, lw=0.6)
            ax.tick_params(labelsize=8)
            if ri == 0:
                ax.set_title(title, fontsize=10, fontweight="bold", color=col)
            if ri < 3:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("Time [s]", fontsize=9)

    # add a note about pitch settling on the trained AoA panel
    td3_al = td3_traj["alpha"]
    ref_al = np.rad2deg(ta["alpha"])
    settled = np.where(np.abs(td3_al - ref_al) < 0.3)[0]
    if len(settled):
        ts = td3_traj["t"][settled[0]]
        axes[1, 1].axvline(ts, color="#777", lw=1., ls=":")
        axes[1, 1].text(ts + 0.8, ref_al + 0.3,
                        f"within 0.3°\nat t≈{ts:.0f}s",
                        fontsize=7.5, color="#555")

    # ---- envelope sweep (column 2, rows 0-3, using rows 0,1,2) ----
    # merge row 3 col 2 is unused
    axes[3, 2].set_visible(False)

    sweep_rows = [
        (0, np.rad2deg(ana["alpha"]),   np.rad2deg(l_al),
         r"AoA $\alpha^*$ [deg]",       "Angle of Attack"),
        (1, np.rad2deg(ana["delta_e"]), np.rad2deg(l_de),
         r"Elevator $\delta_e^*$ [deg]","Elevator Deflection"),
        (2, ana["delta_t"],             l_dt,
         r"Throttle $\delta_t^*$ [–]",  "Throttle Setting"),
    ]
    for ri, y_ana, y_rl, ylabel, title in sweep_rows:
        ax = axes[ri, 2]
        ax.plot(Vg, y_ana, color=CA, lw=1.8, ls="--", label="Analytical trim")
        ax.plot(Vg, y_rl,  color=CB, lw=2.0, marker="o", ms=5, label="Learned (TD3)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.22, lw=0.6)
        ax.tick_params(labelsize=8)
        if ri < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Commanded airspeed [m/s]", fontsize=9)

    axes[0, 2].set_title(
        "Trim Schedule — Learned vs Analytical\n(40–65 m/s envelope)",
        fontsize=10, fontweight="bold")

    fig.text(0.5, 0.008,
             f"Envelope MAE — Elevator: {mae_de:.2f}°    "
             f"AoA: {mae_al:.2f}°    "
             f"Throttle: {mae_dt:.3f}    "
             f"(seed 0, averaged over final 8 s per operating point)",
             ha="center", fontsize=9, style="italic", color="#444")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
