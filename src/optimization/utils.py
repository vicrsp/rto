import numpy as np


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def build_F(t, x):
    F0, tm, Fs, ts, Fmin = [0.002, x[0], x[1], x[2], 0]
    Ft = np.zeros_like(t)
    for index, tstamp in enumerate(t):
        F = F0
        if(tstamp > tm):
            F = Fs
        if(tstamp > ts):
            F = Fmin
        Ft[index] = F

    return Ft


def calculate_SSE(sim_values, samples):
    # SSE
    error = 0
    for time, sim_value in sim_values.items():
        meas_value = samples[time]
        for i in range(len(meas_value)):
            error = error + \
                ((meas_value[i] - sim_value[i])/meas_value[i])**2
    return error


def convert_ivp_results(ivp_results, keys=None):
    results = {}
    time = ivp_results.t
    signals = ivp_results.y
    for i, signal in enumerate(signals):
        key = i if keys == None else keys[i]
        results[key] = np.transpose(np.vstack((time, signal)))

    return results
