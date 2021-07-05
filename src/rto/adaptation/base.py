from abc import ABC, abstractmethod
from bunch import Bunch

class AdaptationStrategy(ABC):
    def __init__(self, process_model, initial_data):
        self.process_model = process_model
        self.initial_data = initial_data

    @abstractmethod
    def adapt(self, u, samples):
        pass

    @abstractmethod
    def get_adaptation(self, u):
        return None


class AdaptationResult:
    def __init__(self, type, values):
      self.type = type
      self.values = values
    
    def get(self):
        return Bunch(self.values)

    