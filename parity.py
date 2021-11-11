"""Module for a very simple parity benchmark."""
from benchmark import *
import numpy 

class Parity(BenchMark):
    """
    Very simple Parity benchmark.
    
    inputs are integers from 0 to size-1, outputs are input%2.
    """

    def __init__(self):
        """Initialise the parity benchmark class."""
        return

    def create_training_set(self, size):
        """Create the training set for the benchmark."""
        keys = [i/size for i in range(size)]
        keys = numpy.random.permutation(keys)
        # index 0 shld be 0 if it is even, 0.5 if it is odd
        # index 1 shld be 0.5 if it is even, 0 if it is odd
        values = [numpy.array([(int(keys[i]*size)%2)/2, (~int(keys[i]*size)%2)/2]) for i in range(size)]
        training_set = {}
        for i in range(size):
            training_set[keys[i]] = values[i]
        return training_set

    def reset(self):
        return
