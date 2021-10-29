"""Creat a NARMA10 dataset"""
import numpy
from NARMA import NARMA

class Narma10(NARMA):
    """Create a NARMA10 dataset."""

    def __init__(self, mode="discretised"):
        """
        Initialise the dataset and set the mode.
        mode (default: "discretised"): whether the output will be acquired in an instantaneous or dicretised fashion.
        """
        super().__init__(mode)
        self.alpha = 0.3
        self.beta = 0.05
        self.gamma = 1.5
        self.delta = 0.1
        self.N = 10
        return