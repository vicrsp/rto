from scipy.optimize import differential_evolution, Bounds
import numpy as np

class ProfileOptimizer:
    # F0, tm, Fm, ts, Fs = x 
    def __init__(self, ub = [0.002,250,0.002,250,0.002], lb=[0.002,0,0,0,0], g = [0.025,0.15]):
        self.g = g
        self.bounds = Bounds(lb, ub)
        self.xk = []
        self.fxk = []
        self.gk = []
        

    def Run(self, model, x0 = [0.002,60,0.001,100,0.0]):

        self.xk.append(x0)
        f0, g0 = model.fobj(x0)
        self.fxk.append(f0)
        self.gk.append(g0)
        
        self.model = model

        results = differential_evolution(func=self.EvalObjective, bounds=self.bounds, args=(model,), 
        disp=False, popsize=10, seed=8754, callback=self.SaveResults)
        return results.fun, results.x, self.xk

    def SaveResults(self, xk, convergence):
    
        self.xk.append(xk)
        f, g = self.model.fobj(xk)
        self.fxk.append(f)
        self.gk.append(g)

    def EvalObjective(self, x, model):
        fx, g = model.fobj(x)
        
        # Penalize solutions that violates constraints
        constraints = np.array(g) - np.array(self.g)
        if(np.any(constraints > 0)):
            fx = fx/1000
        
        fx = 1 / fx

        return fx


class ModelParameterOptimizer:
    def __init__(self, lb=[0.01, 0.01], ub=[0.5,0.5]):
        self.xk = []
        self.fxk = []
        self.bounds = Bounds(lb, ub)

    def Run(self, model, input, samples, x0 = [0.053, 0.128]):
        self.xk.append(x0)
        f0 = model.GetSSE(input, x0, samples)
        self.fxk.append(f0)
        
        self.model = model
        self.samples = samples
        self.input = input

        results = differential_evolution(func=self.EvalObjective, bounds=self.bounds, args=(model,input), 
        disp=False, popsize=50, seed=8754, callback=self.SaveResults)
        return results.fun, results.x, self.xk
    
    def SaveResults(self, xk, convergence):
        self.xk.append(xk)
        self.fxk.append(self.model.GetSSE(self.input, xk, self.samples))

    def EvalObjective(self, x, model, input):
        return model.GetSSE(input, x, self.samples)