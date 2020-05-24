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

## Get optimal profiles for initial set of params
opt = ProfileOptimizer()
model_aprox = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])
f_aprox, x_aprox, results_aprox = opt.run(model_aprox)

## Calibrate model parameters
cal = ModelParameterOptimizer()
model_ideal = SemiBatchReactor()

samples = model_ideal.get_samples(x_aprox, [0.95,0.97,0.99])
f_cal, x_cal, results_cal = cal.run(model_aprox, x_aprox, samples)
print('Fcal---> {}'.format(f_cal))
print('Xcal---> {}'.format(x_cal))

model_adjusted = SemiBatchReactor(k=[x_cal[0], x_cal[1], 0, 0, 5])
sim_adjusted = model_adjusted.simulate(x_aprox)
sim_ideal = model_ideal.simulate(x_aprox)
sim_aprox = model_aprox.simulate(x_aprox)

# Plot the optimal simulation profiles
fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True)

ax1[0].plot(sim_ideal.t, sim_ideal.y[0], 'b')
ax1[0].plot(sim_aprox.t, sim_aprox.y[0], 'r--')
ax1[0].plot(sim_adjusted.t, sim_adjusted.y[0], 'k-.')
ax1[0].plot(np.array(list(samples.keys())) * 250, np.array(list(samples.values()))[:,0], 'go')
ax1[0].legend(['Ca(ideal)','Ca(aprox)','Ca(Adjusted)'])

ax1[1].plot(sim_ideal.t, sim_ideal.y[2], 'b')
ax1[1].plot(sim_aprox.t, sim_aprox.y[2], 'r--')
ax1[1].plot(sim_adjusted.t, sim_adjusted.y[2], 'k-.')
ax1[1].plot(np.array(list(samples.keys())) * 250, np.array(list(samples.values()))[:,2], 'go')
ax1[1].legend(['Cc(ideal)','Cc(aprox)','Cc(Adjusted)'])

ax2[0].plot(sim_ideal.t, sim_ideal.y[1], 'b')
ax2[0].plot(sim_aprox.t, sim_aprox.y[1], 'r--')
ax2[0].plot(sim_adjusted.t, sim_adjusted.y[1], 'k-.')
ax2[0].plot(np.array(list(samples.keys())) * 250, np.array(list(samples.values()))[:,1], 'go')
ax2[0].legend(['Cb(ideal)','Cb(aprox)','Cb(Adjusted)'])

ax2[1].plot(sim_ideal.t, sim_ideal.y[3], 'b')
ax2[1].plot(sim_aprox.t, sim_aprox.y[3], 'r--')
ax2[1].plot(sim_adjusted.t, sim_adjusted.y[3], 'k-.')
ax2[1].plot(np.array(list(samples.keys())) * 250, np.array(list(samples.values()))[:,3], 'go')
ax2[1].legend(['Cd(ideal)','Cd(aprox)','Cd(Adjusted)'])

plt.xlabel('Time [m]')
plt.ylabel('mol L-1')
plt.show()

plt.figure()
plt.plot(sim_ideal.t, sim_ideal.y[4], 'b')
plt.plot(sim_aprox.t, sim_aprox.y[4], 'r--')
plt.plot(sim_adjusted.t, sim_adjusted.y[4], 'k-.')
plt.legend(['V(ideal)','V(aprox)','V(Adjusted)'])
plt.show()


