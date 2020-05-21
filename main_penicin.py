import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def dC_peel_pectin_dt(u, dx, C_peel_proto, k_hydrolysis, D_pectin):
    dydt = np.empty_like(u)

    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dydt[0]    = C_peel_proto * k_hydrolysis + D_pectin * (-2.0*u[0] + 2.0*u[1]) / dx**2
    dydt[1:-1] = C_peel_proto * k_hydrolysis + D_pectin * np.diff(u,2) / dx**2
    dydt[-1]   = C_peel_proto * k_hydrolysis + D_pectin * (-2.0*u[-1] + 2.0*u[-2]) / dx**2

    return dydt


def process_equations(w, t, x, u):

    df = [
        ]

    return df


# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10.0
numpoints = 100

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
u0 = np.random.randn(10)
w0 = [10, 10, 0, 0, 0, 0]
k = np.zeros_like(t)

# Call the ODE solver.
wsol = odeint(process_equations, w0, t, args=(),
              atol=abserr, rtol=relerr)