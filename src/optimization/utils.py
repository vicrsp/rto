import numpy as np


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


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

def calculate_SSE(sim_values, samples):
    # SSE
    error = 0
    for time, sim_value in sim_values.items():
        meas_value = samples[time]
        for i in range(len(meas_value)):
                error = error + \
                    ((meas_value[i] - sim_value[i])/meas_value[i])**2
    return error
