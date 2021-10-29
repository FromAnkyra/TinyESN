"""Module for a very simple parity benchmark."""
from timeseries import *
import numpy as np

class Parity(TimeSeries):
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
        keys = np.random.permutation(keys)
        values = [int(keys[i]*size)%2 for i in range(size)]
        training_set = {}
        for i in range(size):
            training_set[keys[i]] = values[i]
        return training_set

    def reset(self):
        return

