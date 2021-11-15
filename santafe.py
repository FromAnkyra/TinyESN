from benchmark import *
import numpy
"""
Take the santa fe laser time series dataset and turn it into a benchmark.

originally defined at: Engel, Yaakov, Shie Mannor, and Ron Meir. 2004. “The Kernel Recursive Least Squares Algorithm.” IEEE Transactions on Signal Processing: A Publication of the IEEE Signal Processing Society 52 (8): 2275–85.
available at: https://github.com/MaterialMan/CHARC/blob/master/Support%20files/other/Datasets/laser.txt
"""
class SantaFe(BenchMark):
    def __init__(self):
        f = open("datasets/santa-fe-laser.txt", "r")
        lines = f.readlines()
        data_un = [int(line) for line in lines[1:]]
        normaliser = max(data_un)*2
        self.data = [element/normaliser for element in data_un]
        return

    def create_training_set(self, size):
        training_set = {}
        for i in range(len(self.data)-1):
            training_set[(self.data[i])] = numpy.array([self.data[i+1]])
        return training_set

    def reset(self):
        return