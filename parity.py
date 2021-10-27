from timeseries import *
import numpy as np

class Parity(TimeSeries):
    def __init__(self):
        return

    def create_training_set(self, size):
        keys = [i for i in range(size)]
        keys = np.random.permutation(keys)
        values = [keys[i]%2 for i in range(size)]
        training_set = {}
        for i in range(size):
            training_set[keys[i]] = values[i]
        return training_set

