from benchmark import *
import numpy
"""Take the monthly sunspots dataset and turn it into benchmark data.

link: https://machinelearningmastery.com/time-series-datasets-for-machine-learning/
"""
class SunSpots(BenchMark):
    def __init__(self):
        f = open("datasets/monthly-sunspots.csv", "r")
        lines = f.readlines()
        data_un = [float(line.split(",")[1].strip("\n")) for line in lines[1:]]
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
