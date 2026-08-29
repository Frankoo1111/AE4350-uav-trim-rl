"""
validate.py
===========
Validation of the whole project.  Runs a set of independent
checks across the dynamics model, the analytical trim solver, the Gymnasium
environment, the TD3 agent, the trained models, and the training loop, and
prints a PASS/FAIL line for each plus a final score.

Run:  python3 validate.py
"""

from __future__ import annotations
import os, sys, json, glob, warnings
import numpy as np

warnings.filterwarnings("ignore")

from aircraft import AircraftParams, derivatives, rk4_step, aero_forces
from analytical_trim import trim_at, trim_schedule
from trim_env import UAVTrimEnv, EnvConfig
from td3 import TD3, TD3Config, ReplayBuffer
from train import train, evaluate

CHECKS = []
def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco

p = AircraftParams()


# ----------------------------------------------------------------- dynamics
@check("aircraft: derivatives finite, shape (4,)")
def _():
    xd = derivatives(np.array([52., 0., 0., 0.03]), np.array([-0.05, 0.5]), p)
    assert xd.shape == (4,) and np.all(np.isfinite(xd)), xd

@check("aircraft: RK4 step finite & shape-preserving")
def _():
    s = rk4_step(np.array([52., 0., 0., 0.03]), np.array([-0.05, 0.5]), 0.01, p)
    assert s.shape == (4,) and np.all(np.isfinite(s))

@check("aircraft: lift increases with angle of attack")
def _():
    L1 = aero_forces(24, 0.02, 0, 0, 0.3, p)[0]
    L2 = aero_forces(24, 0.10, 0, 0, 0.3, p)[0]
    assert L2 > L1 > 0

@check("aircraft: thrust scales linearly with throttle")
def _():
    T0 = aero_forces(24, 0.05, 0, 0, 0.0, p)[3]
    T1 = aero_forces(24, 0.05, 0, 0, 1.0, p)[3]
    assert abs(T0) < 1e-9 and abs(T1 - p.T_max) < 1e-9


# ----------------------------------------------------------------- analytical trim
@check("trim: converges across full envelope (residual < 1e-6)")
def _():
    sch = trim_schedule(np.linspace(40, 65, 13), p)
    assert np.all(sch["converged"]) and np.max(sch["residual"]) < 1e-6

@check("trim: alpha decreases & throttle increases with airspeed")
def _():
    sch = trim_schedule(np.linspace(40, 65, 13), p)
    assert np.all(np.diff(sch["alpha"]) < 0)
    assert np.all(np.diff(sch["delta_t"]) > 0)

@check("trim: trimmed controls give ~zero accelerations in dynamics")
def _():
    t = trim_at(52.5, p)
    xd = derivatives(np.array([52.5, 0., 0., t["theta"]]),
                     np.array([t["delta_e"], t["delta_t"]]), p)
    assert np.max(np.abs(xd)) < 1e-6, xd


# ----------------------------------------------------------------- environment
@check("env: passes gymnasium env_checker")
def _():
    from gymnasium.utils.env_checker import check_env
    check_env(UAVTrimEnv(p, EnvConfig()), skip_render_check=True)

@check("env: reset obs in observation_space and finite")
def _():
    env = UAVTrimEnv(p, EnvConfig())
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs) and np.all(np.isfinite(obs))

@check("env: 500-step random rollout stays finite, reward scalar")
def _():
    env = UAVTrimEnv(p, EnvConfig())
    env.action_space.seed(0); obs, _ = env.reset(seed=0)
    for _ in range(500):
        o, r, te, tr, i = env.step(env.action_space.sample())
        assert np.all(np.isfinite(o)) and np.isfinite(r)
        assert isinstance(te, bool) and isinstance(tr, bool)
        if te or tr:
            obs, _ = env.reset()

@check("env: out-of-range actions are clipped (no crash)")
def _():
    env = UAVTrimEnv(p, EnvConfig()); env.reset(seed=0)
    o, r, te, tr, i = env.step(np.array([10.0, -10.0]))  # way out of [-1,1]
    assert abs(i["delta_e"]) <= p.de_limit + 1e-9
    assert -1e-9 <= i["delta_t"] <= 1.0 + 1e-9

@check("env: reward at analytical trim ~= alive_bonus")
def _():
    env = UAVTrimEnv(p, EnvConfig()); t = trim_at(52.5, p)
    env.reset(options={"V_cmd": 52.5})
    env.state = np.array([52.5, 0., 0., t["theta"]])
    env.prev_action_phys = np.array([t["delta_e"], t["delta_t"]])
    a = np.array([t["delta_e"] / p.de_limit, 2 * t["delta_t"] - 1])
    _, r, *_ = env.step(a)
    assert abs(r - env.cfg.alive_bonus) < 1e-3, r

@check("env: terminates on departure from envelope")
def _():
    env = UAVTrimEnv(p, EnvConfig()); env.reset(seed=0)
    env.state = np.array([15.0, 0., 0., 0.])   # V far below 28 m/s limit
    _, r, te, tr, i = env.step(np.array([0.0, 0.0]))
    assert te is True

@check("env: truncates exactly at max_steps")
def _():
    env = UAVTrimEnv(p, EnvConfig()); env.reset(seed=1)
    n = 0
    for _ in range(env.max_steps + 5):
        env.state = np.array([24., 0., 0., 0.05])  # keep it inside the envelope
        _, r, te, tr, i = env.step(np.array([0.0, 0.3]))
        n += 1
        if tr:
            break
    assert n == env.max_steps, n

@check("env: reset is deterministic given a seed")
def _():
    e1 = UAVTrimEnv(p, EnvConfig()); e2 = UAVTrimEnv(p, EnvConfig())
    o1, _ = e1.reset(seed=123); o2, _ = e2.reset(seed=123)
    assert np.allclose(o1, o2) and e1.V_cmd == e2.V_cmd


# ----------------------------------------------------------------- TD3 agent
@check("td3: actor outputs within [-1, 1]")
def _():
    ag = TD3(7, 2, TD3Config())
    for _ in range(50):
        a = ag.act(np.random.randn(7), noise=0.0)
        assert np.all(a >= -1 - 1e-6) and np.all(a <= 1 + 1e-6)

@check("td3: replay buffer add/sample/wraparound")
def _():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(7, 2, size=10)
    for k in range(25):  # force wraparound past size=10
        buf.add(np.ones(7) * k, np.zeros(2), 1.0, np.ones(7), 0.0)
    assert buf.size == 10 and buf.ptr == 25 % 10
    o, a, r, no, d = buf.sample(4, rng)
    assert o.shape == (4, 7) and a.shape == (4, 2) and r.shape == (4, 1)

@check("td3: update() runs and returns finite critic loss")
def _():
    ag = TD3(7, 2, TD3Config(batch_size=32))
    rng = np.random.default_rng(0)
    for _ in range(40):
        ag.buffer.add(rng.standard_normal(7), rng.uniform(-1, 1, 2),
                      rng.standard_normal(), rng.standard_normal(7), 0.0)
    closs, aloss = ag.update()
    assert np.isfinite(closs)

@check("td3: save/load reproduces identical actions")
def _():
    ag = TD3(7, 2, TD3Config())
    obs = np.random.randn(7)
    a_before = ag.act(obs, noise=0.0)
    path = "/tmp/_td3_ckpt.pt"; ag.save(path)
    ag2 = TD3(7, 2, TD3Config()); ag2.load(path)
    a_after = ag2.act(obs, noise=0.0)
    assert np.allclose(a_before, a_after, atol=1e-6)

@check("td3: same seed -> identical initial policy")
def _():
    a1 = TD3(7, 2, TD3Config(seed=5)); a2 = TD3(7, 2, TD3Config(seed=5))
    obs = np.random.randn(7)
    assert np.allclose(a1.act(obs), a2.act(obs), atol=1e-7)


# ----------------------------------------------------------------- trained models
@check("models: all 3 trained seeds load")
def _():
    files = glob.glob("runs/main/model_seed*.pt")
    assert len(files) >= 3, files
    for f in files:
        ag = TD3(7, 2, TD3Config()); ag.load(f)

@check("models: seed0 trims envelope (|acc|<0.5, no crash)")
def _():
    env = UAVTrimEnv(p, EnvConfig())
    ag = TD3(7, 2, TD3Config()); ag.load("runs/main/model_seed0.pt")
    ev = evaluate(ag, env, n_episodes=12, seed=777)
    assert ev["ss_accel"] < 0.5, ev      # accelerations near zero = trim found
    assert ev["return_mean"] > 0, ev      # positive return = not crashing

@check("models: learned trim schedule elevator MAE < 3 deg")
def _():
    env = UAVTrimEnv(p, EnvConfig())
    ag = TD3(7, 2, TD3Config()); ag.load("runs/main/model_seed0.pt")
    Vg = np.linspace(40, 65, 9); ana = trim_schedule(Vg, p)
    tail = int(8.0 / env.cfg.dt); learned = []
    for V in Vg:
        env.reset(options={"V_cmd": V})
        env.state = np.array([V, 0., 0., 0.03])
        env.prev_action_phys = np.array([0.0, 0.5])
        obs, _ = env._observe(env.state, env.prev_action_phys)
        des = []
        for t in range(env.max_steps):
            a = ag.act(obs, 0.0); obs, r, te, tr, i = env.step(a)
            if t >= env.max_steps - tail:
                des.append(i["delta_e"])
            if te:
                break
        learned.append(np.mean(des))
    mae = np.rad2deg(np.mean(np.abs(np.array(learned) - ana["delta_e"])))
    assert mae < 3.0, f"elevator MAE {mae:.2f} deg"


# ----------------------------------------------------------------- training loop
@check("train: short run completes, history well-formed")
def _():
    out = train(total_steps=6000, eval_every=3000, n_eval=4,
                td3_cfg=TD3Config(seed=0, start_steps=2000), verbose=False)
    h = out["history"]
    assert len(h["steps"]) == 2
    assert all(np.isfinite(h["eval_return"]))

@check("train: identical seed -> identical learning curve (reproducible)")
def _():
    cfg = lambda: TD3Config(seed=3, start_steps=2000)
    h1 = train(6000, 3000, 4, td3_cfg=cfg(), verbose=False)["history"]
    h2 = train(6000, 3000, 4, td3_cfg=cfg(), verbose=False)["history"]
    assert np.allclose(h1["eval_return"], h2["eval_return"], atol=1e-6), \
        (h1["eval_return"], h2["eval_return"])

@check("train: learning improves return over a longer run")
def _():
    out = train(total_steps=12000, eval_every=4000, n_eval=8,
                td3_cfg=TD3Config(seed=0, start_steps=4000), verbose=False)
    R = out["history"]["eval_return"]
    assert R[-1] > R[0] + 100, R   # clearly learns


# ----------------------------------------------------------------- runner
def run_all():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    passed = 0
    width = max(len(n) for n, _ in CHECKS)
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
            passed += 1
        except Exception as e:
            msg = str(e).splitlines()[0][:80] if str(e) else type(e).__name__
            print(f"  FAIL  {name:<{width}}  -> {msg}", flush=True)
    print("\n" + "=" * 60)
    print(f"  {passed}/{len(CHECKS)} checks passed")
    print("=" * 60)
    return passed == len(CHECKS)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
