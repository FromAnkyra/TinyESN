"""Creat a NARMA10 dataset"""
import numpy

class Narma:
    #TODO: change this so it's just NARMA and it takes the size as an argument
    """Create a NARMA10 dataset."""

    def __init__(self, dimension=10, mode="discretised"):
        """
        Initialise the dataset and set the mode.
        dimension (default: 10): number of dimensions in the NARMA10 benchmark
        mode (default: "discretised"): whether the output will be acquired in an instantaneous or dicretised fashion.
        """
        self.dimension = dimension
        self.y = 0
        self.timestep = 0
        self.history = {}
        self.input = 0
        possible_modes = ["discretised", "instantaneous"]
        if mode not in possible_modes:
            raise ValueError("mode type is invalid.")
        self.mode = mode
        return
        
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
            input_r = 0
            history_sum = 0
        elif len(self.history) < self.dimension-1:
            r = len(self.history)
            input_r = list(self.history.keys())[self.timestep - r]
            for i in range(r):
                history_sum += list(self.history.values())[(self.timestep-1)-i]
        else:
            r = self.dimension
            input_r = list(self.history.keys())[self.timestep - r]
            for i in range(r-1):
                history_sum += list(self.history.values())[(self.timestep-1)-i]

        self.y = 0.3*self.y + (0.05*self.y)*history_sum + 1.5*input_r*self.input + 0.1
        self.input = current_input
        self.history[self.input] = float(self.y)
        self.timestep += 1
        return 

    def _increment_timestep_inst(self, current_input):
        """Get the output for the current input in an instantaneous manner."""
        history_sum = 0
        r = 0
        self.input = current_input
        if len(self.history) == 0:
            r = 0
            input_r = 0
            history_sum = 0
        elif len(self.history) < self.dimension-1:
            r = len(self.history)
            input_r = list(self.history.keys())[self.timestep - r]
            for i in range(r):
                history_sum += list(self.history.values())[(self.timestep-1)-i]
        else:
            r = self.dimension-1
            input_r = list(self.history.keys())[self.timestep - r]
            for i in range(r-1):
                history_sum += list(self.history.values())[(self.timestep-1)-i]

        self.y = 0.3*self.y + (0.05*self.y)*history_sum + 1.5*input_r*self.input + 0.1

        self.history[self.input] = float(self.y)
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
    
    
