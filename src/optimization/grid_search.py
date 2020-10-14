import numpy as np
from .utils import calculate_SSE

CA_INDEX = 0
CB_INDEX = 1
CC_INDEX = 2
CD_INDEX = 3
V_INDEX = 4
class ParameterGridSearch:
    def __init__(self, lb=[0.01, 0.01], ub=[0.5,0.5], N=10):
        self.lb = lb
        self.ub = ub
        self.N = N
        
    def build_grid(self):
        grid = {}
        points = []
        for i in range(len(self.lb)):
            points.append(np.linspace(self.lb[i], self.ub[i], self.N))
        
        nx = len(self.lb)
        for i in range(nx):
            for j in range(nx):
                if(j < i): #only lower diagonal params
                    grid['{}_{}'.format(i,j)] = np.meshgrid(points[i], points[j])
        return grid
    
    def run(self, model, input, samples):
        grid = self.build_grid()

        results = {}
        results_pareto = {}
        for key, value in grid.items():
            x, y = value
            nx = len(x)
            ny = len(y)

            res = np.zeros((nx,ny))
            res_pareto = {}

            for i in range(nx):
                for j in range(ny):    
                   res[i,j] = self.eval([x[i,j], y[i,j]], model, input, samples) 
                   res_pareto['{},{}'.format(i,j)] = self.eval_pareto([x[i,j], y[i,j]], model, input, samples)

            results[key] = res
            results_pareto[key] = res_pareto
            
        return grid, results, results_pareto

    def eval(self, x, model, input, samples):
        sim_values = model.get_simulated_samples(input, x, samples)
        sim_values_to_use = {}
        samples_to_use = {}
        for key in samples.keys():
            sim_values_to_use[key] = [sim_values[key][CB_INDEX],
                                      sim_values[key][CC_INDEX], sim_values[key][CD_INDEX]]
            samples_to_use[key] = [samples[key][CB_INDEX],
                                   samples[key][CC_INDEX], samples[key][CD_INDEX]]
        # SSE
        return calculate_SSE(sim_values_to_use, samples_to_use)

    def eval_pareto(self, x, model, input, samples):
        sim_values = model.get_simulated_samples(input, x, samples)
        # Weight vector
        w = np.ones_like(list(samples.values())[0])

        # SSE
        error = np.zeros_like(w)
        for time, sim_value in sim_values.items():
            meas_value = samples[time]
            for i in range(len(meas_value)):
                error[i] = error[i] + w[i]*((meas_value[i] - sim_value[i])/meas_value[i])**2

        return error

    def unpack_errors(self, pareto):
        # Unpack the data for each ELEMENT
        Z_Ca = np.zeros((self.N,self.N))
        Z_Cb = np.zeros((self.N,self.N))
        Z_Cc = np.zeros((self.N,self.N))
        Z_Cd = np.zeros((self.N,self.N))
        Z_V = np.zeros((self.N,self.N))

        for index, errors in pareto.items(): 
            i,j = list(map(int,index.split(",")))
            Z_Ca[i,j] = errors[0]
            Z_Cb[i,j] = errors[1]
            Z_Cc[i,j] = errors[2]
            Z_Cd[i,j] = errors[3]
            Z_V[i,j] = errors[4]    
        
        return [Z_Ca, Z_Cb, Z_Cc, Z_Cd, Z_V]