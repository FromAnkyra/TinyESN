from abc import ABC, abstractmethod

class BenchMark(ABC):
    @abstractmethod
    def create_training_set(self, size):
        pass

    @abstractmethod
    def reset(self):
        pass