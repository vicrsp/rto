import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint, solve_ivp
# from optimization.utils import find_nearest_idx

# add the parent folder to path
# lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
# sys.path.append(lib_path)


# Fixed system parameters
LP_steam = 0.8 # [bar]
MP_steam = 10 # [bar]
HP_steam = 80 # [bar]
TGCondenser_pressure = 0.35 # [bar]
Juice_flow = 500 # [t/h]
Brix_initial = 15 # [ºC]
Brix_final = 65 # [ºC]
Evaporator_Area = 2000 # [m^2]
Turbine_efficiency = 50 #[%]
Turbine_power = 500 # [kW]
Boiler_Steam_Temperature = 650 # [ºK]

HP_Enthalpy_Saturated_Water = 1317.08 # [kJ/kg] at 80Bar
HP_Enthalpy_Saturated_Steam = 2758.61 # [kJ/kg]

MP_Enthalpy_Saturated_Water = 762.683 # [kJ/kg] at 10Bar
MP_Enthalpy_Saturated_Steam = 2777.12 # [kJ/kg]

LP_Enthalpy_Saturated_Water = 145.119 # [kJ/kg] at 0.8Bar
LP_Enthalpy_Saturated_Steam = 2563.93 # [kJ/kg]


class Boiler:
    def __init__(self, f_efficiency):
        self.eta = 0.6
        self.BD = 0.02
        self.f_efficiency = f_efficiency

        pass

    def calculate_enthalpy(self, T, P):
        pass

    def bar_to_mpa(self, x):
        return x / 10
    
    def mpa_to_bar(self, x):
        return x * 10

    """[summary]
        Returns the boiler outlet temperature and pressure
    """    
    def simulate(self, F_i, H_To, HP_o):
        F_o = (1 - self.BD) * F_i
        P_i = LP_steam
        P_o = HP_steam
        pass

class SimpleHeatEnergySystem:
    def __init__(self, lb = [80, 55, 20], ub = [120, 90, 50]):
      self.lb = lb
      self.ub = ub

    def plant_samples(self, x):
        F_1, F_2 = x
        F_3 = F_1 + F_2

        Q_1 = 10000/(85 - 20*np.exp(-(F_1 - 80)/190))
        Q_2 = 65 - 0.006*(F_2 - 80)**2 - 24/F_2
        Q_3 = 80 - 0.006*(F_3 - 35)**2 - 12/F_3

        return [Q_1, Q_2, Q_3]

    def model_samples(self, x):
        F_1, F_2 = x
        F_3 = F_1 + F_2

        Q_1 = 10000/(58 - 0.08*F_1) 
        Q_2 = 65 - 0.06*F_2
        Q_3 = 50 + 0.04*F_3

        return [Q_1, Q_2, Q_3]

    def plant_cost(self, X):
        cost = []
        for x in X:
            Q1, Q2, Q3 = self.plant_samples(x)
            penalty = 5 * np.linalg.norm(x - np.array([90, 60]))
            cost.append(Q1 - Q2 - Q3 + penalty)
        return cost
    
    def model_cost(self, X):
        cost = []
        for x in X:
            Q1, Q2, Q3 = self.model_samples(x)
            penalty = 5 * np.linalg.norm(x - np.array([90, 60]))
            cost.append(Q1 - Q2 - Q3 + penalty)
        return cost

    def grid_search_plot(self, grid_size=100):
        F_1 = np.linspace(self.lb[0], self.ub[0], grid_size)
        F_2 = np.linspace(self.lb[1], self.ub[1], grid_size)

        xx, yy = np.meshgrid(F_1, F_2)
        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        
        # horizontal stack vectors to create x1,x2 input for the model
        X = np.hstack((r1,r2))
        
        cost_plant = np.array(self.plant_cost(X))
        cost_model = np.array(self.model_cost(X))

        # reshape the predictions back into a grid
        cost_plant = cost_plant.reshape(xx.shape)
        cost_model = cost_model.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(8,6))
        CS = ax.contour(xx, yy, cost_plant, cmap='jet', levels=50)
        ax.clabel(CS, inline=True, fontsize=10)
        CS =  ax.contourf(xx, yy, cost_model)
        #ax.clabel(CS, inline=True, fontsize=10)     

        fig.show()

md = SimpleHeatEnergySystem()
md.grid_search_plot()

# class SugarEthanolEnergySystem:
#     def __init__(self, y0=[0.72, 0.05, 0.08, 0.01, 1.0], k=[0.053, 0.128, 0.028, 0.001, 5]):
#         self.y0 = y0
#         self.k1, self.k2, self.k3, self.k4, self.Cb_in = k
#         self.stoptime = 250
#         self.numpoints = 100
#         self.g = [0.025, 0.15]

#         self.boilers = 2
#         self.turbogenerators = 2
#         self.deaerators = 1
#         self.distillation_columns = 1
#         self.evaporators = 5


#     def set_parameters(self, k1, k2):
#         self.k1 = k1
#         self.k2 = k2

#     def get_samples(self, inputs, when, noise=True):
#         sim_results = self.simulate(inputs)

#         samples = {}
#         for value in when:
#             sample = []
#             idx = find_nearest_idx(sim_results.t, value*self.stoptime)
#             for i, result in enumerate(sim_results.y):
#                 val = result[idx]
#                 if(noise == True):
#                     sample.append(val + np.random.normal(0, 0.05*val))
#                 else:
#                     sample.append(val)

#             samples[value] = sample

#         return samples

#     def get_simulated_samples(self, input, x, samples):
#         k1, k2 = x
#         self.set_parameters(k1, k2)
#         sim_results = self.simulate(input)

#         sim_values = {}
#         for time in samples.keys():
#             idx = find_nearest_idx(sim_results.t, time*self.stoptime)
#             sim_value = []
#             for i, result in enumerate(sim_results.y):
#                 sim_value.append(result[idx])

#             sim_values[time] = sim_value

#         return sim_values

#     def odecallback(self, t, w, x):
#         Ca, Cb, Cc, Cd, V = w
#         F0, tm, Fs, ts, Fmin = x
#         F = F0
#         if(t > tm):
#             F = Fs

#         if(t > ts):
#             F = Fmin

#         # variable
#         k1 = self.k1
#         k2 = self.k2
#         k3 = self.k3
#         k4 = self.k4
#         Cb_in = 5

#         # Process model
#         df = [
#             -k1*Ca*Cb - F*Ca/V,
#             -k1*Ca*Cb - 2*k2*Cb*Cb - k3*Cb - k4*Cb*Cc + F*(Cb_in - Cb)/V,
#             k1*Ca*Cb - k4*Cb*Cc - F*Cc/V,
#             k2*Cb*Cb - F*Cd/V,
#             F]

#         return df

#     def simulate(self, x):
#         t = [self.stoptime * float(i) / (self.numpoints - 1)
#              for i in range(self.numpoints)]
#         xnew = [0.002, x[0], x[1], x[2], 0]
#         return solve_ivp(fun=self.odecallback, t_span=[0, self.stoptime], t_eval=t, y0=self.y0, args=(xnew,))

#     def get_objective(self, sim_results, noise=None):
#         Cc_tf = sim_results.y[2][-1]
#         V_tf = sim_results.y[4][-1]
#         fx = Cc_tf * V_tf
#         return -fx if noise == None else -fx * (1 + np.random.normal(scale=noise))

#     def get_constraints(self, x, sim_results, noise=None):
#         Cb_tf = sim_results.y[1][-1]
#         Cd_tf = sim_results.y[3][-1]

#         if(noise != None):
#             Cb_tf = Cb_tf * (1 + np.random.normal(scale=noise))
#             Cd_tf = Cd_tf * (1 + np.random.normal(scale=noise))

#         return np.asarray([Cb_tf, Cd_tf])
