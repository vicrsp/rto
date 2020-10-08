import numpy as np
import json
from scipy.integrate import odeint, solve_ivp
from optimization.utils import find_nearest_idx

CA_INDEX = 0
CB_INDEX = 1
CC_INDEX = 2
CD_INDEX = 3
V_INDEX = 4

class SemiBatchReactor:
    def __init__(self, y0 = [0.72, 0.05, 0.08, 0.01, 1.0], k = [0.053, 0.128, 0.028, 0.001, 5]):
        self.y0 = y0
        self.k1, self.k2, self.k3, self.k4, self.Cb_in = k
        self.stoptime = 250
        self.numpoints = 100

    def set_parameters(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def get_samples(self, inputs, when, noise = True):
        sim_results = self.simulate(inputs)

        samples = {}        
        for value in when:
            sample = []
            idx = find_nearest_idx(sim_results.t, value*self.stoptime)
            for i, result in enumerate(sim_results.y):
                val = result[idx]
                if(noise == True):
                    sample.append(val + np.random.normal(0,0.1*val))
                else:
                    sample.append(val)

            samples[value] = sample

        return samples

    def get_simulated_samples(self, input, x, samples):
        k1, k2 = x
        self.set_parameters(k1, k2)
        sim_results = self.simulate(input)

        sim_values = {}
        for time in samples.keys():
            idx = find_nearest_idx(sim_results.t, time*self.stoptime)
            sim_value = []
            for i, result in enumerate(sim_results.y):
                sim_value.append(result[idx])
            
            sim_values[time] = sim_value
        
        return sim_values

    def odecallback(self, t, w, x):
        Ca, Cb, Cc, Cd, V = w
        F0, tm, Fm, ts, Fs = x 
        F = F0
        if(t > tm):
            F = Fm
        
        if(t > ts):
            F = Fs

        # variable
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4
        Cb_in = 5

        # Process model
        df = [
        -k1*Ca*Cb - F*Ca/V,
        -k1*Ca*Cb -2*k2*Cb*Cb -k3*Cb -k4*Cb*Cc + F*(Cb_in - Cb)/V,
         k1*Ca*Cb - k4*Cb*Cc - F*Cc/V,
         k2*Cb*Cb - F*Cd/V,
         F]

        return df

    def simulate(self, x):
        t = [self.stoptime * float(i) / (self.numpoints - 1) for i in range(self.numpoints)]
        return solve_ivp(fun=self.odecallback, t_span=[0, self.stoptime], t_eval=t, y0=self.y0, args=(x,))