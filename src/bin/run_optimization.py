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

def build_F(t, x):
    F0, tm, Fm, ts, Fs = x 
    Ft = np.zeros_like(t)
    for index, tstamp in enumerate(t):
        F = F0
        if(tstamp > tm):
            F = Fm
        if(tstamp > ts):
            F = Fs
        Ft[index] = F
    
    return Ft
        

opt = ProfileOptimizer()
cal = ModelParameterOptimizer()

model_ideal = SemiBatchReactor()
model_aprox = SemiBatchReactor(k=[0.053, 0.128, 0.0, 0.0, 5])

f_ideal, x_ideal, results_ideal = opt.Run(model_ideal)
f_aprox, x_aprox, results_aprox = opt.Run(model_aprox)

print('Fobj---> Ideal: {}, Approx: {}'.format(1/f_ideal, 1/f_aprox))
print('Xopt---> \n Ideal: {} \n Approx: {}'.format(x_ideal, x_aprox))

# Simulate the solution found
sim_ideal = model_ideal.Simulate(x_ideal)
sim_aprox = model_aprox.Simulate(x_aprox)

# Build the input signal F
F_ideal = build_F(sim_ideal.t, x_ideal)
F_aprox = build_F(sim_aprox.t, x_aprox)

# # Plot the optimal simulation profiles
# fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True)

# ax1[0].plot(sim_ideal.t, sim_ideal.y[0], 'b')
# ax1[0].plot(sim_aprox.t, sim_aprox.y[0], 'r--')
# ax1[0].legend(['Ca(ideal)','Ca(aprox)'])

# ax1[1].plot(sim_ideal.t, sim_ideal.y[2], 'b')
# ax1[1].plot(sim_aprox.t, sim_aprox.y[2], 'r--')
# ax1[1].legend(['Cc(ideal)','Cc(aprox)'])

# ax2[0].plot(sim_ideal.t, sim_ideal.y[1], 'b')
# ax2[0].plot(sim_aprox.t, sim_aprox.y[1], 'r--')
# ax2[0].legend(['Cb(ideal)','Cb(aprox)'])

# ax2[1].plot(sim_ideal.t, sim_ideal.y[3], 'b')
# ax2[1].plot(sim_aprox.t, sim_aprox.y[3], 'r--')
# ax2[1].legend(['Cd(ideal)','Cd(aprox)'])

# plt.xlabel('Time [m]')
# plt.ylabel('mol L-1')

# plt.figure()
# plt.plot(sim_ideal.t, sim_ideal.y[4], 'b')
# plt.plot(sim_aprox.t, sim_aprox.y[4], 'r--')
# plt.legend(['V(ideal)','V(aprox)'])
# #plt.show()

# plt.figure()
# plt.plot(sim_ideal.t, F_ideal, 'b')
# plt.plot(sim_aprox.t, F_aprox, 'r--')
# plt.legend(['Ft(ideal)','Ft(aprox)'])
# #plt.show()

## Calibrate model parameters
samples = model_ideal.GetSamples(x_aprox, [0.95,0.97,0.99])
f_cal, x_cal, results_cal = cal.Run(model_aprox, x_aprox, samples)
print('Fcal---> {}'.format(f_cal))
print('Xcal---> {}'.format(x_cal))

model_adjusted = SemiBatchReactor(k=[x_cal[0], x_cal[1], 0, 0, 5])
sim_adjusted = model_adjusted.Simulate(x_aprox)
sim_ideal = model_ideal.Simulate(x_aprox)

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


