import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# add the parent folder to path
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)

from optimization.model_optimization import ProfileOptimizer, ModelParameterOptimizer
from model.semi_batch import SemiBatchReactor
from optimization.utils import build_F


opt = ProfileOptimizer(
    ub=[0.002, 250, 0.002, 250, 0.002], lb=[0.0, 0, 0, 0, 0])

model_ideal = SemiBatchReactor()
model_aprox = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])

f_ideal, x_ideal, xk_ideal, fxk_ideal, gk_ideal = opt.run(model_ideal)
f_aprox, x_aprox, xk_aprox, fxk_aprox, gk_aprox = opt.run(model_aprox)

print('Fobj---> Ideal: {}, Approx: {}'.format(-f_ideal, -f_aprox))
print('Xopt---> \n Ideal: {} \n Approx: {}'.format(x_ideal, x_aprox))

# Simulate the solution found
sim_ideal = model_ideal.simulate(x_ideal)
sim_aprox = model_aprox.simulate(x_aprox)

# Build the input signal F
F_ideal = build_F(sim_ideal.t, x_ideal)
F_aprox = build_F(sim_aprox.t, x_aprox)

# Plot the optimal simulation profiles
fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True)

ax1[0].plot(sim_ideal.t, sim_ideal.y[0], 'b')
ax1[0].plot(sim_aprox.t, sim_aprox.y[0], 'r--')
ax1[0].legend(['Ca(ideal)', 'Ca(aprox)'])

ax1[1].plot(sim_ideal.t, sim_ideal.y[2], 'b')
ax1[1].plot(sim_aprox.t, sim_aprox.y[2], 'r--')
ax1[1].legend(['Cc(ideal)', 'Cc(aprox)'])

ax2[0].plot(sim_ideal.t, sim_ideal.y[1], 'b')
ax2[0].plot(sim_aprox.t, sim_aprox.y[1], 'r--')
ax2[0].legend(['Cb(ideal)', 'Cb(aprox)'])

ax2[1].plot(sim_ideal.t, sim_ideal.y[3], 'b')
ax2[1].plot(sim_aprox.t, sim_aprox.y[3], 'r--')
ax2[1].legend(['Cd(ideal)', 'Cd(aprox)'])

plt.xlabel('Time [m]')
plt.ylabel('mol L-1')

plt.figure()
plt.plot(sim_ideal.t, sim_ideal.y[4], 'b')
plt.plot(sim_aprox.t, sim_aprox.y[4], 'r--')
plt.legend(['V(ideal)', 'V(aprox)'])

plt.figure()
plt.plot(sim_ideal.t, F_ideal, 'b')
plt.plot(sim_aprox.t, F_aprox, 'r--')
plt.legend(['Ft(ideal)', 'Ft(aprox)'])
plt.show()
