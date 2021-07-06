from abc import ABC, abstractmethod

from scipy.optimize.minpack import fsolve

class ProcessModel(ABC):
    def __init__(self, name, y0, k):
      self.name = name
      self.k = k
      self.y0 = y0
    
    @abstractmethod
    def get_objective(self, x, noise=None):
      raise NotImplementedError
    
    @abstractmethod
    def get_constraints(self, x, noise=None):
      raise NotImplementedError

    @abstractmethod
    def odecallback(self, t, w, x):
      raise NotImplementedError

    def set_parameters(self, k):
      self.k = k

    def solve_steady_state(self, u):  
      def fobj(x):
        return self.odecallback(0, x, u)
      return fsolve(func=fobj, x0=self.y0)
