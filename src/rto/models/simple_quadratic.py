import numpy as np
from abc import abstractmethod
from .base import ProcessModel


class SimpleQuadraticBenchmark(ProcessModel):
    """_summary_

    Args:
        ProcessModel (_type_): _description_
    """
    def __init__(self, name='SimpleQuadraticBenchmark', k=[1, 2]):
        super().__init__(name, None, k)

    def get_objective(self, x, noise=None):
        fx = x[0]**2 + x[1]**2 + self.k[0]*x[0]*x[1]
        return fx if noise is None else fx + np.random.normal(scale=noise)

    def get_constraints(self, x, noise=None):
        g = np.asarray([1 - x[0] + x[1]**2 + self.k[1] * x[1]])
        if(noise is not None):
            return g + np.random.normal(scale=noise)
        return g

    def odecallback(self, t, w, x):
        pass