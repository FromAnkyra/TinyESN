import numpy as numpy

class TinyESN():
    def __init__(self, K: int, N: int, L: int, f):
        self.u = numpy.random.rand(K)
        self.x = numpy.zeros((N), dtype=numpy.float32)
        self.v = numpy.zeros((L), dtype=numpy.float32)
        self.Wu = numpy.random.rand(N, K)
        self.W = numpy.random.rand(N, N)
        self.Wv = numpy.random.rand(N, L)
        self.Wback = numpy.random.rand(N, L)
        self.func = numpy.vectorize(f)

        self.timestep = 0

    def increment_timestep_no_fb_discretised(self):
        #currently using discretised ESN equations for a physical system (Stepney, 2021)
        new_x = self.func(self.W @ self.x + self.Wu @ self.u)
        self.v = self.func(self.x @ self.Wv)  
        self.x = new_x

    def increment_timestep_fb_discretised(self):
        # incorporates feedback of the output
        new_x = self.func(self.W @ self.x + self.Wu @ self.u + self.Wback @ self.v)
        self.v = self.func(self.x @ self.Wv)  
        self.x = new_x
    
    def pretty_print(self):
        print(self.u)
        print(self.x)
        print(self.v)


