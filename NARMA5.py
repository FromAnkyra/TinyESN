"""Create a NARMA5 dataset"""
import numpy
from NARMA import NARMA

class Narma5(NARMA):
    """Create a NARMA5 dataset."""

    def __init__(self, mode="discretised"):
        """
        Initialise the dataset and set the mode.
        mode (default: "discretised"): whether the output will be acquired in an instantaneous or dicretised fashion.
        """
        super().__init__(mode)
        self.alpha = 0.3
        self.beta = 0.05
        self.delta = 5
        self.gamma = 0.1
        return
    