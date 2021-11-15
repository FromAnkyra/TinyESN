from numpy.core.fromnumeric import size
from benchmark import *
import numpy

class Xor_benchmark(BenchMark):
    def __init__(self):
        return

    def create_training_set(self, set_size):
        training_set = {}
        size_root = int(numpy.sqrt(set_size))
        set_size = size_root**2 # resize the thing to be a square number 
        a = numpy.random.randint(size_root, size=set_size)
        a = numpy.random.permutation(a)
        b = numpy.random.randint(size_root, size=set_size)
        b = numpy.random.permutation(b)

        for i in range(size_root):
            for j in range(size_root):
                training_set[(a[i]/(size_root*2), b[j]/(size_root*2))] = numpy.array([(a[i] ^ b[j])/(size_root*2)])
        # ok, this doesn't work, because obviously you cannot repeat keys
        # except we have few enough values that it doesn't actually become anything
        # I think i have "good code"ed myself into a corner
        return training_set

    def reset(self):
        # uhhhhhhhhhhhhhh
        return