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