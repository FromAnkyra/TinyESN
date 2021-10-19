import numpy as numpy
import decimal

class TinyESN():
    def __init__(self, K: int, N: int, L: int, f=numpy.tanh, mode="discretised", feedback=False, topology="random", connectivity=0.1): #note: default topology is fully connected
        #Matrix shapes are taken from [1]
        modes = ["discretised", "instantaneous"]
        if mode not in modes:
            raise ValueError("please set mode to discretised or instantaneous")
        self.mode = mode
        topologies = ["random", "ring", "lattice", "torus", "fully_connected"]
        if topology not in topologies:
            raise ValueError("please set a valid topology")
        self.topology = topology
        self.connectivity = connectivity #this only comes into play if the topology is set to random
        self.feedback = feedback
        self.u = numpy.random.rand(K, 1)
        self.x = numpy.zeros((N, 1), dtype=numpy.float32)
        self.v = numpy.zeros((L, 1), dtype=numpy.float32)
        self.Wu = numpy.random.rand(N, K)
        self.W = numpy.random.zeros((N, N), dtype=numpy.float32)
        self.Wv = numpy.random.rand(L, N)
        self.Wback = numpy.random.rand(N, L)
        self.func = numpy.vectorize(f)
        self.t = 0
        self.M_train = None
        self.D_train = None
        self.outputs = None
        self.timestep = None
        self.set_topology()
        return

    def increment_timestep(self, data):
        if self.mode == "discretised" and self.feedback == True:
            self.increment_timestep_fb_discretised(data)
        elif self.mode == "discretised" and self.feedback == False:
            self.increment_timestep_no_fb_discretised(data)
        elif self.mode == "instantaneous" and self.feedback == True:
            self.increment_fb_inst(data)
        else:
            self.increment_timestep_no_fb_inst(data)
        return

    def set_topology(self):
        if self.topology == "random":
            self.init_random()
        elif self.topology == "ring":
            self.init_ring()
        elif self.topology == "lattice":
            self.init_lattice() #TODO: this is mildly broken
        elif self.topology == "torus":
            self.init_torus() #TODO: this is mildy broken
        elif self.topology == "fully_connected":
            self.init_complete()
        return
    
    def init_random(self):
        #sets topoplogy according randomly 
        
        return
    
    '''sets the reservoir topology to one of standard ones, as described in [2]'''
    
    '''Question: is connectivity vs architecture: if there are two esns with the same architecture and size, are they automatically the same connectivity? Or is it a case of the architecture describes _which_ nodes can have things happen to them and then the connectivity is how many of those nodes are actually connected?'''
    def init_ring(self):
        placeholder = numpy.zeros((self.x.size, self.x.size))
        for i in range(self.x.size):
            coordinates = [(i, i, i), ((i-1)%self.x.size, i, (i+1)%self.x.size)]
            placeholder[tuple(coordinates)] = 1
        self.W = placeholder
        # "Each node has a single self-connection and a weighted connection to each of its neighbours to its left and right" ([2])
        return
    
    def init_lattice(self, rows):
        #TODO: test
        if self.x.size % rows == 0:
            raise ValueError("this would not let you form a grid")
        cols = self.x.size / rows
        s = self.x.size
        placeholder = numpy.zeros(s, s)
        for i in range(self.x.size):
            bot = max(i-(rows+1), 0) - 1 # this is ugly as fuck but is the best way i can think of doing it
            #this isn't gonna work, i need to take a break and come back to it
            top = min(i+(rows+1), s) - 1
            coordinates = [(i, i, i, i, i, i, i, i, i), 
                            (bot-1, bot, bot+1, 
                            max(i-1, 0), i, min(i+1, s), 
                            top-min(i, 1), top, top+1)]
            placeholder[tuple(coordinates)] = 1
        
        self.W = placeholder
        # "With this topology, we define a square grid of neurons each connected to its nearest neighbours (using its Moore neighbourhood, as commonly used in cellular automata). Each non-perimetre node has eight connections and one self-connection, resulting with each node having a maximum of nine adaptable weights in W" ([2])
        return

    def init_torus(self, rows):
        #TODO: test
        if self.x.size % rows == 0:
            raise ValueError("this would not let you form a grid")
        cols = self.x.size / rows
        s = self.x.size
        placeholder = numpy.zeros(s, s)
        for i in range(self.x.size):
            bot = (i - rows)%s # this is ugly as fuck but is the best way i can think of doing it
            top = (i + rows)%s
            coordinates = [(i, i, i, i, i, i, i, i, i), 
                            (bot-1, bot, bot+1, 
                            (i-1)%s, i, i+1%s, 
                            top-1, top, top+1)]
            placeholder[tuple(coordinates)] = 1
        
        self.W = placeholder
        #"The torus topology is a special case of the latice where the perimetre nodes are connected to give periodic boundary conditions. Each node has nine adaptable weights in W" ([2])
        return

    def init_complete(self):

        #"The fully connected network has no topological constraints  and has the maximum number of adaptable weights: the weight matrix W is fully populated." ([2])
        #As this is the default setting for the generation anyway, simply need to return
        return
    
    def init_sparse(self, connectivity: decimal.Decimal):
        return

    '''Checks to see whether the ESN has the Echo State property [1]'''
    def has_echo_state(self):
        #calculate spectral radius
        #check the "largest singular value"
        return False

    '''updates the reservoir according  to the equations laid out in [1]'''
    def increment_timestep_no_fb_inst(self, input):
        self.u = input
        self.x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u)) 
        self.v = self.func(numpy.dot(self.Wv, self.x))  
        return
    
    def increment_fb_inst(self, input):
        self.u = input
        self.x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u) + numpy.dot(self.Wback, self.v))
        self.v = self.func(numpy.dot(self.x, self.Wv))  
    
    '''updates the reservoir state and outputs according to the non-instantaneous equation described in [3], adapted from the initial definition from [1]'''
    
    def increment_timestep_no_fb_discretised(self, input):
        #currently using discretised ESN equations for a physical system (Stepney, 2021)
        new_x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u)) 
        # print(f"Wv: {self.Wv.shape},\nx: {self.x.shape}")
        self.v = self.func(numpy.dot(self.Wv, self.x))  
        self.x = new_x
        self.u = input
        self.t += 1
        return

    def increment_timestep_fb_discretised(self, input):
        # incorporates feedback of the output
        new_x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u) + numpy.dot(self.Wback, self.v))
        self.v = self.func(numpy.dot(self.x, self.Wv))  
        self.x = new_x
        self.u = input
        self.t += 1
        return
    
    def pretty_print(self):
        print(self.u)
        print(self.x)
        print(self.v)
        return


    '''trains the matrix using the matrix pseudoinverse method described in [4]
    
    Note: [4] describes this method as "it is expensive memory-wise for large design matrices [self.x]". That being said, given the scale of this ESN, it is perhaps the most straightforward option, as it is very simple to understand.'''
    def train_pseudoinverse(self, training_set):
        #TODO: turn M into the correct shape
        for data in training_set:
            self.increment_timestep(data)
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
                # print(f"M = {self.M_train.shape},\n D={self.D_train.shape},\n M+={numpy.linalg.pinv(self.M_train).shape}")
                self.Wv = numpy.transpose(numpy.dot(numpy.linalg.pinv(self.M_train), self.D_train)) #Note: the output matrix derived with this method gives a transposition of the weight matrix described in [1], hence the transposition here
        return

    '''Trains the ESN using linear regression using the method described in [4].'''

    # def train_linear_regression(self, training_set, regularisation_coefficient):
    #     return
    
    '''tests the ESN '''
    def test(self, testing_set):
        self.increment_timestep(list(testing_set.keys())[0])
        self.outputs = self.v
        for data in list(testing_set.keys())[1:]:
            self.increment_timestep(data)
            self.outputs = numpy.vstack((self.outputs, self.v))
        return 



'''
Bibliography:

[1] Jaeger, Herbert. n.d. “The ‘echo State’ Approach to Analysing and Training Recurrent Neural Networks – with an Erratum note1.” Accessed October 5, 2021. http://www.faculty.jacobs-university.de/hjaeger/pubs/EchoStatesTechRep.pdf.

[2] Dale, Matthew, Simon O’Keefe, Angelika Sebald, Susan Stepney, and Martin A. Trefzer. 2021. “Reservoir Computing Quality: Connectivity and Topology.” Natural Computing 20 (2): 205–16.

[3] Stepney, Susan. n.d. “Non-Instantaneous Information Transfer in Physical Reservoir Computing.

[4] Montavon, Grégoire, Geneviève B. Orr, and Klaus-Robert Müller, eds. 2012. Neural Networks: Tricks of the Trade: Second Edition. Springer, Berlin, Heidelberg.
'''
            





