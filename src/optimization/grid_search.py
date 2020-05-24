import numpy as np

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
        for key, value in grid.items():
            x, y = value
            nx = len(x)
            ny = len(y)

            res = np.zeros((nx,ny))
            for i in range(nx):
                for j in range(ny):    
                   res[i,j] = model.GetSSE(input, [x[i,j], y[i,j]], samples) 
                   print('Iter: {}-{}/{}-{}'.format(i,j,nx,ny))

            results[key] = res
            

        return grid, results


