# Reinforcement Learning of Fixed-Wing UAV Trim Conditions

AE4350 — Bio-inspired Intelligence and Learning for Aerospace Applications.

A reinforcement-learning agent learns the **trim schedule** of a fixed-wing
UAV: for any commanded airspeed it discovers the control-surface deflection and
thrust setting that produce **steady, wings-level flight** (zero accelerations).
This replaces the tedious manual trim look-up tables used in practice with a
single learned feedback policy.

The setup follows the supervisor's guidance for the approved proposal:

> "You can let the agent set control deflections and give rewards based on
>  resulting accelerations."

Moreover, this links nicely to my internship topic conducted at Saab in the fall of 2026

---

## 1. Problem formulation

**Plant.** A 4-state / 2-input nonlinear longitudinal model of a
small UAV (`aircraft.py`), integrated with RK4.

| symbol | meaning | symbol | meaning |
|--------|---------|--------|---------|
| `V`    | airspeed `[m/s]`        | `delta_e` | elevator `[rad]`  |
| `gamma`| flight-path angle `[rad]` | `delta_t` | throttle `[0..1]` |
| `q`    | pitch rate `[rad/s]`    |           |                   |
| `theta`| pitch attitude `[rad]`  | `alpha = theta - gamma` | angle of attack |

**RL task (`trim_env.py`, a `gymnasium.Env`).** Each episode draws a random
commanded airspeed `V_cmd` in `[18, 30] m/s`. The agent applies control
deflections; the aircraft is integrated forward; the agent must reach and hold
trim at `V_cmd`.

* **Action** `a in [-1,1]^2` -> `(delta_e, delta_t)` (scaled internally).
* **Observation** (7-d): airspeed error, `alpha`, `q`, `gamma`, the command
  `V_cmd`, and the *resulting* accelerations `V_dot`, `q_dot`.
* **Reward** = `alive_bonus - cost`, where the cost is dominated by the
  **resulting accelerations** `(V_dot, V*gamma_dot, q_dot)`, exactly zero at
  trim, plus light terms that select *which* trim is wanted (airspeed-to-command
  error and level-flight `gamma`). All terms are scaled to be O(1) and the cost
  is clipped for stable learning.

**Algorithm (`td3.py`).** TD3 (Twin Delayed DDPG, Fujimoto et al. 2018),
implemented from scratch in PyTorch: deterministic actor, twin critics with
clipped double-Q targets, target-policy smoothing, and delayed actor/target
updates. A deterministic policy is the natural choice because trim is a
deterministic target (one correct control vector per airspeed).

---

## 2. Files

```
aircraft.py          UAV longitudinal dynamics + RK4 integrator
analytical_trim.py   fsolve-based trim solver (ground-truth reference)
trim_env.py          Gymnasium environment (action = deflections, reward = accelerations)
td3.py               TD3 agent (networks, replay buffer, training step)
train.py             training loop + evaluation; importable train()  and  CLI
analyze.py           multi-seed training + figures 1-3
sensitivity.py       hyperparameter sweeps + figure 4
validate.py          26-check end-to-end validation harness
requirements.txt
figures/             generated PNGs
runs/                saved models and learning histories
```

## 3. Install & run

```bash
pip install -r requirements.txt

# validate the whole project (26 checks: dynamics, env, agent, models, training)
python3 validate.py

# sanity check: analytical trim schedule
python3 analytical_trim.py

# single training run (saves model + history)
python3 train.py --steps 50000 --seed 0 --outdir runs/default

# main results: train 3 seeds and make figs 1-3
python3 analyze.py --train-seeds 3 --steps 50000

# sensitivity analysis (figure 4); full sweep = 4 hyperparameters
python3 sensitivity.py --run --seeds 3 --steps 35000
# or a subset:
python3 sensitivity.py --run --seeds 3 --steps 35000 --params learning_rate gamma
```

On a single CPU a 50k-step run takes a few minutes; convergence is reached
around ~25k steps. A GPU is not necessarily required.

---

## 4. Results (included in `figures/`)

* **`fig1_learning_curve.png`** — evaluation return and steady-state trim
  airspeed error vs. training steps, mean ± std over seeds. **The learning
  effect:** return climbs from strongly negative to near the ~320 maximum and
  the airspeed error falls from several m/s to < 0.6 m/s.
* **`fig2_trim_schedule.png`** — the **learned** trim controls vs. the
  **analytical** ground truth across the airspeed envelope. Agreement:
  mean abs. error ≈ **0.5° elevator, 0.2° angle of attack, 0.013 throttle** —
  the agent rediscovered the correct trim physics without ever being told it.
* **`fig3_time_histories.png`** — closed-loop response from an off-trim initial
  condition at `V_cmd = 20` and `28 m/s`; all states/controls settle onto the
  analytical trim values within ~2-4 s.
* **`fig4_sensitivity.png`** — final performance vs. learning rate and discount
  factor (mean ± std). Higher learning rate learns faster within the budget;
  the discount factor shows the classic inverted-U with `gamma = 0.99` best.

> Note: the bundled `fig4` was produced with a short 18k-step / 2-seed budget to
> keep it cheap to regenerate; re-run `sensitivity.py` with the defaults
> (`--seeds 3 --steps 35000`) for the full version and add the `expl_noise` and
> `hidden` sweeps.

---

## 5. Mapping to the assessment rubric

| rubric item | where it is addressed |
|---|---|
| Complexity of the method | TD3 (state-of-the-art), **continuous** state & action |
| Environment complexity | continuous nonlinear 4-state flight dynamics, full airspeed envelope |
| Learning effect | `fig1` learning curve |
| Sensitivity analysis | `fig4`, multiple hyperparameters, multiple seeds |
| Description of results | statistics over seeds (mean ± std) in `fig1`, `fig4` |
| Analysis of the found solution | `fig2`/`fig3`: learned policy matches analytical trim |
| Reproducibility | seeded; open-source code; every variable documented |

## 6. Notes / possible extensions

* Add steady climb/descent by commanding `gamma != 0` (already supported by the
  reward structure — just feed a non-zero target).
* Extend to the lateral-directional axes for full 6-DOF trim.
* Swap TD3 for SAC (stochastic) and compare sample efficiency.
* Maybe increase the fidelity of the model