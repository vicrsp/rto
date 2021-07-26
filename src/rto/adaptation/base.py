from abc import ABC, abstractmethod
from bunch import Bunch
import numpy as np

class AdaptationStrategy(ABC):
    def __init__(self, process_model, initial_data, type, ub, lb):
        self.process_model = process_model
        self.initial_data = initial_data
        self.type = type
        self.lb = np.array(lb)
        self.ub = np.array(ub)

    def normalize_input(self, X):
        return (X - self.lb)/(self.ub - self.lb)
    
    def denormalize_input(self, X):
        return X * (self.ub - self.lb) + self.lb

    def get_model_parameters(self):
        return self.process_model.initial_parameters

    @abstractmethod
    def adapt(self, u, samples):
        pass

    @abstractmethod
    def get_adaptation(self, u):
        return None

    def update_operating_point(self, u, samples):
        return u


class AdaptationResult:
    def __init__(self, values):
      self.values = values
    
    def get(self):
        return Bunch(self.values)

    