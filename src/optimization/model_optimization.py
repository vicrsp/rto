from optimization.utils import build_F
from scipy.optimize import differential_evolution, Bounds
import numpy as np

class ProfileOptimizer:
    # F0, tm, Fm, ts, Fs = x 
    def __init__(self, ub = [0.002,250,0.002,250,0.002], lb=[0.0,0,0,0,0], g = [0.025,0.15]):
        self.g = g
        self.bounds = Bounds(lb, ub)
        self.xk = []
        self.fxk = []
        self.gk = []
        

    def run(self, model, x0 = [0.002,60,0.001,100,0.0]):

        self.xk.append(x0)
        f0 = self.eval_objective(x0, model)
        self.fxk.append(f0)
        #self.gk.append(g0)
        
        self.model = model

        results = differential_evolution(func=self.eval_objective, bounds=self.bounds, args=(model,), 
        disp=False, popsize=10, seed=8754, callback=self.save_results)
        return results.fun, results.x, self.xk

    def run_obj_b(self, model, x0 = [0.002,60,0.001,100,0.0]):

        self.xk.append(x0)
        f0 = self.eval_objective_b(x0, model)
        self.fxk.append(f0)
        #self.gk.append(g0)

        self.model = model

        results = differential_evolution(func=self.eval_objective_b, bounds=self.bounds, args=(model,), 
        disp=True, popsize=20, tol=0.001, seed=8754, callback=self.save_results)
        return results.fun, results.x, self.xk

    def save_results(self, xk, convergence):
        self.xk.append(xk)
        f = self.eval_objective(xk, self.model)
        self.fxk.append(f)
        #self.gk.append(g)

    def eval_objective(self, x, model):
        sim_results = model.simulate(x)
        Cb_tf = sim_results.y[1][-1]
        Cc_tf = sim_results.y[2][-1]
        Cd_tf = sim_results.y[3][-1]
        V_tf = sim_results.y[4][-1]

        fx = -Cc_tf * V_tf
        g = [Cb_tf, Cd_tf]
        
        # Penalize solutions that violates constraints
        constraints = np.array(g) - np.array(self.g)
        if(np.any(constraints > 0)):
            fx = fx + 1000
        
        #fx = 1 / fx

        return fx

    def eval_objective_b(self, x, model):
        sim_results = model.simulate(x)
        
        Cb_tf = sim_results.y[1][-1]
        Cc_tf = sim_results.y[2][-1]
        Cd_tf = sim_results.y[3][-1]
        V_tf = sim_results.y[4][-1]
        g = [Cb_tf, Cd_tf]

        F = np.array(build_F(sim_results.t, x))
        w = 2500 #penalty

        #Objective
        fx = -(Cc_tf * V_tf - w*(F @ np.transpose(F)))

        # Penalize solutions that violates constraints
        constraints = np.array(g) - np.array(self.g)
        if(np.any(constraints > 0)):
            fx = fx + 1000

        return fx


class ModelParameterOptimizer:
    def __init__(self, lb=[0.0011, 0.0026], ub=[0.212,0.5120]):
        self.xk = []
        self.fxk = []
        self.bounds = Bounds(lb, ub)

    def run(self, model, input, samples, x0 = [0.053, 0.128]):
        
        self.model = model
        self.samples = samples
        self.input = input

        self.xk.append(x0)
        f0 = self.eval_objective(x0, model, input)
        self.fxk.append(f0)
        
        results = differential_evolution(func=self.eval_objective, bounds=self.bounds, args=(model,input), 
        disp=False, popsize=50, seed=8754, callback=self.save_results)
        return results.fun, results.x, self.xk
    
    def save_results(self, xk, convergence):
        self.xk.append(xk)
        self.fxk.append(self.eval_objective(xk, self.model, self.input))

    def eval_objective(self, x, model, input):
        sim_values = model.get_simulated_samples(input, x, self.samples)
        # Weight vector
        w = np.ones_like(input)

        # SSE
        error = 0
        for time, sim_value in sim_values.items():
            meas_value = self.samples[time]
            for i in range(len(meas_value)):
                if(i > 0 & i < 4):
                    error = error + w[i]*((meas_value[i] - sim_value[i])/meas_value[i])**2
        return error