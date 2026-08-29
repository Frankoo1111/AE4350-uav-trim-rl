# aircraft.py
# -----------
# Simulates the physics of a 1000 kg fixed-wing UAV.
#
# The aircraft state is 4 numbers:
#   V     = airspeed [m/s]
#   gamma = flight-path angle [rad]  (positive = climbing)
#   q     = pitch rate [rad/s]
#   theta = pitch attitude [rad]
#
# The agent controls 2 things:
#   delta_e = elevator deflection [rad]  (-25 to +25 degrees)
#   delta_t = throttle [0 to 1]

import numpy as np


# All the aircraft numbers in one place.
# These are fixed for the whole simulation
class AircraftParams:
    def __init__(self):
        # mass and geometry
        self.m   = 1000.0   # mass [kg]
        self.Iyy = 4500.0   # pitch moment of inertia [kg m^2]
        self.S   = 17.0     # wing area [m^2]
        self.b   = 14.3     # span [m]
        self.rho = 1.225    # air density at sea level [kg/m^3]
        self.g   = 9.81     # gravity [m/s^2]

        # engine
        self.T_max = 1500.0   # max thrust [N]

        # aerodynamic coefficients
        self.CL0      =  0.28   # lift at zero AoA
        self.CL_alpha =  5.50   # lift slope [1/rad]
        self.CL_q     =  8.00   # pitch-rate effect on lift
        self.CL_de    =  0.25   # elevator effect on lift

        self.CD0      =  0.025  # zero-lift drag
        self.oswald_e =  0.85   # wing efficiency

        self.Cm0      =  0.020  # pitch moment at zero AoA
        self.Cm_alpha = -1.80   # negative = statically stable
        self.Cm_q     = -25.0   # negative = pitch damping
        self.Cm_de    = -1.10   # elevator pitch effectiveness

        # limits
        self.de_limit    = np.deg2rad(25.0)   # max elevator = 25 degrees
        self.alpha_stall = np.deg2rad(14.0)   # stall angle = 14 degrees

        # derived values
        self.c         = self.S / self.b                              # mean chord [m]
        self.AR        = self.b**2 / self.S                           # aspect ratio
        self.k_induced = 1.0 / (np.pi * self.oswald_e * self.AR)     # induced drag factor


def aero_forces(V, alpha, q, delta_e, delta_t, p):
    """
    Calculate lift, drag, pitching moment, and thrust.
    Returns: L, D, M, T  (all in SI units)
    """
    # dynamic pressure
    q_bar = 0.5 * p.rho * V * V

    # non-dimensional pitch rate (standard aerodynamics normalization)
    q_hat = q * p.c / (2.0 * max(V, 0.001))

    # aerodynamic coefficients
    CL = p.CL0 + p.CL_alpha * alpha + p.CL_q * q_hat + p.CL_de * delta_e
    CD = p.CD0 + p.k_induced * CL**2
    Cm = p.Cm0 + p.Cm_alpha * alpha + p.Cm_q * q_hat + p.Cm_de * delta_e

    # forces and moments
    L = q_bar * p.S * CL          # lift [N]
    D = q_bar * p.S * CD          # drag [N]
    M = q_bar * p.S * p.c * Cm    # pitching moment [N m]
    T = delta_t * p.T_max         # thrust [N]

    return L, D, M, T


def derivatives(state, control, p):
    """
    How fast is each state variable changing right now?
    This is just F = ma applied to the aircraft.
    Returns: [V_dot, gamma_dot, q_dot, theta_dot]
    """
    V, gamma, q, theta = state
    delta_e, delta_t   = control
    alpha = theta - gamma    # angle of attack

    L, D, M, T = aero_forces(max(V, 0.001), alpha, q, delta_e, delta_t, p)

    V_dot     = (T * np.cos(alpha) - D) / p.m  -  p.g * np.sin(gamma)
    gamma_dot = (T * np.sin(alpha) + L - p.m * p.g * np.cos(gamma)) / (p.m * max(V, 0.001))
    q_dot     = M / p.Iyy   # pitch angular acceleration
    theta_dot = q            # attitude changes at pitch rate

    return np.array([V_dot, gamma_dot, q_dot, theta_dot])


def rk4_step(state, control, dt, p):
    """
    Move the simulation forward by one time step (dt seconds).
    Uses Runge-Kutta 4, which is more accurate than simple Euler.
    """
    k1 = derivatives(state,               control, p)
    k2 = derivatives(state + 0.5*dt*k1,  control, p)
    k3 = derivatives(state + 0.5*dt*k2,  control, p)
    k4 = derivatives(state + dt*k3,      control, p)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
