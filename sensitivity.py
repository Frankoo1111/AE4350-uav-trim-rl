"""
sensitivity.py
==============
Sensitivity analysis of the learning algorithm to its main hyperparameters

For each hyperparameter we sweep a grid of values, train `--seeds` independent
agents per value, and report the final-performance mean +/- std.  Results are
cached to sensitivity.json so figures can be re-drawn without re-training.

Swept hyperparameters (all are *learning-algorithm* parameters):
    actor/critic learning rate, discount factor gamma,
    exploration-noise std, hidden-layer width.

Usage
-----
  python3 sensitivity.py --run --seeds 3 --steps 35000
  python3 sensitivity.py            # just (re)plot from cache
"""

from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import train
from td3 import TD3Config

CACHE = "sensitivity.json"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# (label, attribute, grid of values)
SWEEPS = {
    "learning_rate": ("learning rate", [1e-4, 3e-4, 1e-3]),
    "gamma":         ("discount factor", [0.95, 0.99, 0.999]),
    "expl_noise":    ("exploration noise std", [0.05, 0.1, 0.2]),
    "hidden":        ("hidden layer width", [64, 128, 256]),
}


def cfg_for(param, value, seed):
    c = TD3Config(seed=seed, start_steps=5000)
    if param == "learning_rate":
        c.actor_lr = c.critic_lr = value
    elif param == "gamma":
        c.gamma = value
    elif param == "expl_noise":
        c.expl_noise = value
    elif param == "hidden":
        c.hidden_sizes = (int(value), int(value))
    return c


def final_metric(history):
    """Average of the last two evaluation returns (robust final performance)."""
    return float(np.mean(history["eval_return"][-2:]))


def run(seeds, steps, only=None):
    results = {}
    if os.path.exists(CACHE):
        results = json.load(open(CACHE))
    sweeps = SWEEPS if not only else {k: SWEEPS[k] for k in only}
    for param, (label, values) in sweeps.items():
        results.setdefault(param, {"label": label, "values": [], "data": {}})
        for v in values:
            key = str(v)
            returns = results[param]["data"].get(key, [])
            for s in range(len(returns), seeds):     # resume from where stopped
                print(f"[{param}={v}] seed {s} ...", flush=True)
                out = train(total_steps=steps, eval_every=2500, n_eval=8,
                            td3_cfg=cfg_for(param, v, s), verbose=False)
                returns.append(final_metric(out["history"]))
                results[param]["data"][key] = returns
                if v not in results[param]["values"]:
                    results[param]["values"].append(v)
                json.dump(results, open(CACHE, "w"), indent=2)  # checkpoint per seed
    return results


def plot():
    if not os.path.exists(CACHE):
        print("no sensitivity.json; run with --run first")
        return
    results = json.load(open(CACHE))
    params = [p for p in SWEEPS if p in results and results[p]["data"]]
    n = len(params)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, param in zip(axs[0], params):
        r = results[param]
        vals = sorted(r["data"].keys(), key=lambda x: float(x))
        x = [float(v) for v in vals]
        means = [np.mean(r["data"][v]) for v in vals]
        stds = [np.std(r["data"][v]) for v in vals]
        ax.errorbar(range(len(x)), means, yerr=stds, marker="o", capsize=4,
                    lw=2, color="C0")
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_xlabel(r["label"]); ax.set_ylabel("final return")
        ax.set_title(r["label"]); ax.grid(alpha=0.3)
    fig.suptitle("Sensitivity of final performance to learning hyperparameters "
                 "(mean +/- std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_sensitivity.png"), dpi=140)
    print("saved fig4_sensitivity.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=35000)
    ap.add_argument("--params", nargs="*", default=None,
                    help="subset of: learning_rate gamma expl_noise hidden")
    args = ap.parse_args()
    if args.run:
        run(args.seeds, args.steps, only=args.params)
    plot()


if __name__ == "__main__":
    main()
