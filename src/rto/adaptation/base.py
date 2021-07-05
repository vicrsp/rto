from abc import ABC, abstractmethod
from bunch import Bunch

class AdaptationStrategy(ABC):
    def __init__(self, process_model, initial_data, type):
        self.process_model = process_model
        self.initial_data = initial_data
        self.type = type

    @abstractmethod
    def adapt(self, u, samples):
        pass

    @abstractmethod
    def get_adaptation(self, u):
        return None


class AdaptationResult:
    def __init__(self, values):
      self.values = values
    
    def get(self):
        return Bunch(self.values)

    