import numpy as numpy

class TinyESN():
    def __init__(self, K: int, N: int, L: int, f):
        self.u = numpy.random.rand(N, 1)
        self.x = numpy.zeros((N, 1), dtype=numpy.float32)
        self.v = numpy.zeros((N, 1), dtype=numpy.float32)
        self.Wu = numpy.random.rand(N, K)
        self.W = numpy.random.rand(N, N)
        self.Wv = numpy.random.rand(L, N)
        self.Wback = numpy.random.rand(N, L)
        self.func = numpy.vectorize(f)
        self.t = 0
        self.M_train = None
        self.D_train = None
        self.outputs = None
        self.timestep = None
        return

    def increment_timestep_no_fb_discretised(self):
        #currently using discretised ESN equations for a physical system (Stepney, 2021)
        new_x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u)) 
        # print(f"Wv: {self.Wv.shape},\nx: {self.x.shape}")
        self.v = self.func(numpy.dot(self.Wv, self.x))  
        self.x = new_x
        self.t += 1
        return

    def increment_timestep_fb_discretised(self):
        # incorporates feedback of the output
        new_x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u) + numpy.dot(self.Wback, self.v))
        self.v = self.func(numpy.dot(self.x, self.Wv))  
        self.x = new_x
        self.t += 1
        return
    
    def pretty_print(self):
        print(self.u)
        print(self.x)
        print(self.v)
        return

    def train_pseudoinverse(self, training_set):
        #TODO: turn M into the correct shape
        for data in training_set:
            self.u = data
            # print(self.t)
            self.increment_timestep_no_fb_discretised()
            if self.t <= 10:
                pass
            elif self.M_train is None:
                self.M_train = self.x # note: here i am going by the assumption that numpy is doign assign by value, not by reference, which as far as I can tell is true
                # print(f"M (pre-reshape): {self.M_train.shape}")
                self.M_train = numpy.transpose(self.M_train)
                self.D_train = numpy.array([training_set[data]])
                self.outputs = self.v
                # print(f"M = {self.M_train.shape},\n D={self.D_train.shape},\n M+={numpy.linalg.pinv(self.M_train).shape}")
                self.Wv = numpy.dot(numpy.linalg.pinv(self.M_train), self.D_train)
            else:
                y = self.x
                y = numpy.transpose(y)
                self.M_train = numpy.vstack((self.M_train, y))
                self.D_train = numpy.vstack((self.D_train, training_set[data]))
                self.outputs = numpy.vstack((self.outputs, self.v))
                # so what i think is somehow going wrong here is that 
                # print(f"M = {self.M_train.shape},\n D={self.D_train.shape},\n M+={numpy.linalg.pinv(self.M_train).shape}")
                self.Wv = numpy.transpose(numpy.dot(numpy.linalg.pinv(self.M_train), self.D_train)) #Not sure if this is OK - may be worth bringing up
        return
    
    def test(self, testing_set):
        self.u = list(testing_set.keys())[0]
        self.increment_timestep_no_fb_discretised()
        self.outputs = self.v
        for data in list(testing_set.keys())[1:]:
            self.u = data
            self.increment_timestep_no_fb_discretised()
            self.outputs = numpy.vstack((self.outputs, self.v))
        return 




            





