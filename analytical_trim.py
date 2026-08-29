# analytical_trim.py
# ------------------
# Finds the exact trim solution by solving the equilibrium equations.
# This is the "correct answer" that the RL agent is trying to learn.
#
# At trim: V_dot = 0, gamma_dot = 0, q_dot = 0
# That gives 3 equations with 3 unknowns: alpha, delta_e, delta_t
# We solve them numerically with scipy.

import numpy as np
from scipy.optimize import fsolve
from aircraft import AircraftParams, aero_forces


def trim_at(V, p):
    """
    Find the trim controls at a given airspeed V.
    Returns a dictionary with alpha, delta_e, delta_t, and theta.
    """

    def equations(z):
        # z = [alpha, delta_e, delta_t]  -- the three unknowns
        alpha, delta_e, delta_t = z
        L, D, M, T = aero_forces(V, alpha, 0.0, delta_e, delta_t, p)

        # at trim, all three of these must be zero
        eq1 = T * np.cos(alpha) - D          # no acceleration along velocity
        eq2 = T * np.sin(alpha) + L - p.m * p.g   # lift = weight
        eq3 = M                              # no pitch acceleration

        return [eq1, eq2, eq3]

    # starting guess for the solver
    initial_guess = [0.05, 0.0, 0.3]

    # solve the equations
    solution, _, converged, _ = fsolve(equations, initial_guess, full_output=True)
    alpha, delta_e, delta_t = solution

    return {
        "V":       V,
        "alpha":   alpha,
        "delta_e": delta_e,
        "delta_t": delta_t,
        "theta":   alpha,    # at trim: theta = alpha (because gamma = 0)
        "converged": converged == 1,
    }


def trim_schedule(V_list, p):
    """
    Compute trim for a whole list of airspeeds.
    Returns a dict of arrays (one value per airspeed).
    """
    results = [trim_at(V, p) for V in V_list]

    return {
        "V":       np.array([r["V"]       for r in results]),
        "alpha":   np.array([r["alpha"]   for r in results]),
        "delta_e": np.array([r["delta_e"] for r in results]),
        "delta_t": np.array([r["delta_t"] for r in results]),
        "theta":   np.array([r["theta"]   for r in results]),
        "converged": np.array([r["converged"] for r in results]),
    }


# Run this file directly to print the trim schedule
if __name__ == "__main__":
    p = AircraftParams()
    print(f"{'V [m/s]':>8}  {'alpha [deg]':>11}  {'elevator [deg]':>14}  {'throttle':>8}")
    print("-" * 50)
    for V in [40, 45, 50, 55, 60, 65]:
        t = trim_at(V, p)
        print(f"{V:8.1f}  {np.rad2deg(t['alpha']):11.2f}  "
              f"{np.rad2deg(t['delta_e']):14.2f}  {t['delta_t']:8.3f}")
