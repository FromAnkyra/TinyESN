"""Abstract module for the various NARMA benchmarks."""
from abc import abstractmethod
import numpy
from benchmark import BenchMark

class NARMA(BenchMark):
    """Abstract class for the NARMA benchmarks."""

    @abstractmethod
    def __init__(self, mode="discretised"):
        """Initialise the various NARMA benchmarks. Set the various parametres to zero."""
        self.y = 0
        self.timestep = 0
        self.history = {}
        self.input = 0
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.delta = 0
        self.N = 0
        possible_modes = ["discretised", "instantaneous"]
        if mode not in possible_modes:
            raise ValueError("mode type is invalid.")
        self.mode = mode
    
    def increment_timestep(self, current_input):
        """Increment the timestep based on the mode chosen."""
        if self.mode == "discretised":
            self._increment_timestep_discretised(current_input)
        else:
            self._increment_timestep_inst(current_input)
        return 

    def _increment_timestep_discretised(self, current_input):
        """Get the output for the current input in a discretised manner."""
        history_sum = 0
        r = 0
        
        if len(self.history) == 0:
            r = 0
            input_N = 0
            history_sum = 0
        elif len(self.history) < self.N-1:
            r = len(self.history)
            input_N = list(self.history.keys())[self.timestep - r]
            for i in range(r):
                history_sum += list(self.history.values())[(self.timestep-1)-i]
        else:
            r = self.N-1
            input_N = list(self.history.keys())[self.timestep - r]
            for i in range(r-1):
                history_sum += list(self.history.values())[(self.timestep-1)-i]

        self.y = self.alpha*self.y + (self.beta*self.y)*history_sum + self.gamma*input_N*self.input + self.delta
        self.input = current_input
        self.history[self.input] = numpy.array([float(self.y)])
        self.timestep += 1
        return 

    def _increment_timestep_inst(self, current_input):
        """Get the output for the current input in an instantaneous manner."""
        history_sum = 0
        r = 0
        self.input = current_input
        if len(self.history) == 0:
            r = 0
            input_N = 0
            history_sum = 0
        elif len(self.history) < self.N - 1:
            r = len(self.history)
            input_N = list(self.history.keys())[self.timestep - r]
            for i in range(r):
                history_sum += list(self.history.values())[(self.timestep-1)-i]
        else:
            r = self.N - 1
            input_N = list(self.history.keys())[self.timestep - r]
            for i in range(r-1):
                history_sum += list(self.history.values())[(self.timestep-1)-i]

        self.y = self.alpha*self.y + (self.beta*self.y)*history_sum + self.gamma*input_N*self.input + self.delta

        self.history[self.input] = numpy.array([float(self.y)])
        self.timestep += 1

    def _generate_input(self):
        """Generate a random input."""
        return numpy.random.random_sample()/2
    
    def create_training_set(self, size):
        """
        Create a set of NARMA10 inputs and outputs.
        
        size: size of the output set.
        """
        for _ in range(size):
            self.increment_timestep(self._generate_input()) 
        return self.history

    def reset(self):
        self.history = {}
        self.timestep = 0
        return
