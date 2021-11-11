import numpy as numpy
import decimal
import igraph

class TinyESN():
    """
    ESN implementation.

    class that implements ESNs with various parametres, topologies, and update styles.
    Initialising, training, and testing the ESN are all supported.
    currently training is only done using the pseudoinverse method.
    """
    def __init__(self, K: int, N: int, L: int, f=numpy.tanh, mode="discretised", feedback=False, topology="random", connectivity=0.1, input_norm=True):
        """
        Initialise the ESN.

        K: number of input nodes.
        N: number of reservoir nodes.
        L: number output nodes.
        f (default numpy.tanh): update function
        mode (default "discretised"): update mode. values: "discretised", "immediate".
        feedback (default False): whether feedback is accounted for when updating.
        topology (default random): topology of the reservoir layer. values: "random", "ring", "lattice", "torus", "fully_connected".
        connectivity (default 0.1): connectivity of the weight matrix (only relevant for the random topology).
        """
        #Matrix shapes are taken from [1]
        self.input_norm = input_norm
        modes = ["discretised", "instantaneous"]
        if mode not in modes:
            raise ValueError("please set mode to discretised or instantaneous")
        self.mode = mode
        topologies = ["random", "ring", "lattice", "torus", "fully_connected"]
        if topology not in topologies:
            raise ValueError("please set a valid topology")
        self.topology = topology
        if connectivity > 1 or connectivity < 0:
            raise ValueError("please set a connectivity between 0 and 1.")
        self.connectivity = connectivity #this only comes into play if the topology is set to random
        self.feedback = feedback
        self.u = numpy.random.rand(K, 1)
        self.x = numpy.zeros((N, 1), dtype=numpy.float32)
        self.v = numpy.zeros((L, 1), dtype=numpy.float32)
        self.Wu = numpy.random.uniform(low=-1.0, high=1.0, size=N*K)
        self.Wu = self.Wu.reshape((N, K)) #TODO: play around with this
        self.W = numpy.zeros((N, N), dtype=numpy.float32)
        self.Wv = numpy.random.rand(L, N)
        self.Wback = numpy.random.rand(N, L)
        self.func = numpy.vectorize(f)
        self.t = 0
        self.M_train = None
        self.D_train = None
        self.outputs = None
        self.timestep = None
        self._set_topology()
        self._scale_weights()
        return


    def _scale_weights(self):
        """Scale weight matrix to get its spectral radius to 1"""
        absolute = numpy.vectorize(lambda a: abs(a))
        eigen_vals = numpy.linalg.eigvals(self.W)
        spectral_radius = max(absolute(eigen_vals))
        if spectral_radius not in eigen_vals:
            spectral_radius = -spectral_radius
        self.W = self.W * (1/spectral_radius)
        return
    
    def _set_topology(self):
        """Initialise the topology accoding to what is set in init."""
        if self.topology == "random":
            self._init_random()
        elif self.topology == "ring":
            self._init_ring()
        elif self.topology == "lattice":
            self._init_lattice() #TODO: this is mildly broken
        elif self.topology == "torus":
            self._init_torus() #TODO: this is mildy broken
        elif self.topology == "fully_connected":
            self._init_complete()
        return
    
    def _init_random(self):
        """Set reservoir to a random architecture with connectivity equal to that set in init."""
        #sets topology according randomly 
        size = self.x.size
        no_of_weights = int(self.connectivity * (size**2))
        weights = numpy.random.uniform(low=-1.0, high=1.0, size=no_of_weights)
        placeholder = numpy.append(numpy.zeros(size**2 - no_of_weights), weights)
        placeholder = numpy.random.permutation(placeholder).reshape((size, size))
        self.W = placeholder
        return
    
    def _init_ring(self):
        """
        Initialise the ESN with a ring topology.

        From [2]:  Each node has a single self-connection and a weighted connection to each of its neighbours to its left and right.
        
        This imitates the topology of "delay-line reservoirs".
        """
        #TODO: "sparsely connected ring"
        placeholder = numpy.zeros((self.x.size, self.x.size))
        for i in range(self.x.size):
            coordinates = [(i, i, i), ((i-1)%self.x.size, i, (i+1)%self.x.size)]
            placeholder[tuple(coordinates)] = 1
        self.W = placeholder
        return
    
    def _init_lattice(self):
        """
        Initialise the ESN with a lattice topology. ]

        For this to work, the number of nodes N must be a square number, or else the function will return an error.

        From [2]: With this topology, we define a square grid of neurons each connected to its nearest neighbours (using its Moore neighbourhood, as commonly used in cellular automata). Each non-perimetre node has eight connections and one self-connection, resulting with each node having a maximum of nine adaptable weights in W.

        This topology aims to imitate the layout of physical materia.
        """
        if numpy.sqrt(self.x.size) != int(numpy.sqrt(self.x.size)):
            raise ValueError("Can only form a lattice if nodes can form a square.")
        s = self.x.size
        side = int(numpy.sqrt(s))
        placeholder = numpy.zeros((s, s))
        edge_big = lambda a, b, c: a if a<c else b
        edge_small = lambda a, b, c: a if a>=c else b
        col_count = 0
        row_count = 1
        for i in range(s):
            if col_count == side and row_count!=5:
                col_count = 0
                row_count+=1
            col_count+=1
            #this awful (affectionate) piece of code is the moore neighbourhood for any given node i
            #i pity the person to try and modify this to work as a rectangle (i am so sorry)
            coordinates = [(i, edge_big(i+side, i, s), edge_big(i+1, i, (side*row_count)), edge_big((i+1+side), i, min(s, (side*(row_count+1)))), edge_small(i-side, i, 0), edge_small(i-1, i, side*(row_count-1)), edge_small(i-side-1, i, max(0, side*(row_count-2))), edge_small(edge_big(i-side+1, i, side*(row_count-1)), i, 0), edge_big(edge_small(i+side-1, i, side*(row_count)),i,s)), (i, i, i, i, i, i, i, i, i)]
            # print(edge_big(i-side+1, i, side*(row_count-1)))
            # print(coordinates)
            placeholder[tuple(coordinates)] = 1
        
        self.W = placeholder
        return

    def _init_torus(self):
        """
        Initialise the ESN with a lattice topology.

        For this to work, the number of nodes N must be a square number, or else the function will return an error. 

        From [2]: The torus topology is a special case of the latice where the perimetre nodes are connected to give periodic boundary conditions. Each node has nine adaptable weights in W.
        """
        if numpy.sqrt(self.x.size) != int(numpy.sqrt(self.x.size)):
            raise ValueError("Can only form a torus if nodes can form a square.")
        s = self.x.size
        side = int(numpy.sqrt(s))
        placeholder = numpy.zeros((s, s))
        edge_big = lambda a, b, c: a if a<c else b
        edge_small = lambda a, b, c: a if a>=c else b
        col_count = 0
        row_count = 1
        for i in range(s):
            if col_count == side and row_count!=side :
                col_count = 0
                row_count+=1
            col_count+=1
            # you cannot hate me more than i hate myself
            #i pity the person to try and modify this to work as a rectangle (i am so sorry)
            coordinates = [(i, (i+side)%s, edge_big(i+1, i-(side-1), row_count*side), int(edge_big((i+1+side), (i+1), ((row_count+1)*side)))%s, i-side, edge_small(i-1, (row_count*side)-1, side*(row_count-1)), edge_small(i-side-1, i-1, (row_count-2)*side), edge_big(i-side+1, (row_count-2)*side, side*(row_count-1)), edge_small(i+side-1, side*(row_count+1)-1, side*(row_count))%s), (i, i, i, i, i, i, i, i, i)]
            # print(edge_big(i-side+1, i, side*(row_count-1)))
            # print(coordinates)
            placeholder[tuple(coordinates)] = 1
        
        self.W = placeholder
        return

    def _init_complete(self):
        """Initialise an ESN with a fully-connected topology."""
        placeholder = numpy.ones_like(self.W)
        self.W = placeholder
        return

    def has_echo_state(self):
        """Look at the values of the spectral radius and largest singular value to guess if the ESN has the echo state property[1]."""
        absolute = numpy.vectorize(abs)
        abs_eigen_vals = absolute(numpy.linalg.eig(self.W))
        spectral_radius = max(abs_eigen_vals)
        single_vals = numpy.linalg.svd(self.W)
        lsv = max(single_vals)
        #calculate spectral radius
        #check the "largest singular value"
        print(f"spectral radius: {spectral_radius}\nlargest singular value:{lsv}")
        return lsv > 1 and spectral_radius < 1

    def _increment_timestep(self, data):
        """Increment timestep according to the protocol chosen in init."""
        if self.input_norm:
            data = numpy.tanh(data)
        if self.mode == "discretised" and self.feedback is True:
            self._increment_timestep_fb_discretised(data)
        elif self.mode == "discretised" and self.feedback is False:
            self._increment_timestep_no_fb_discretised(data)
        elif self.mode == "instantaneous" and self.feedback is True:
            self._increment_fb_inst(data)
        else:
            self._increment_timestep_no_fb_inst(data)
        return

    def _increment_timestep_no_fb_inst(self, data):
        """Update the ESN state and outputs according to the instantaneous equation defined in [1]. Does not take feedback into account."""
        self.u = data
        self.x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u)) 
        self.v = self.func(numpy.dot(self.Wv, self.x))  
        self.t += 1
        return
    
    def _increment_fb_inst(self, data):
        """Update the ESN state and outputs according to the instantaneous equation defined in [1]. Takes feedback into account."""
        self.u = data
        self.x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u) + numpy.dot(self.Wback, self.v))
        self.v = self.func(numpy.dot(self.x, self.Wv))  
        self.t += 1
        return
    
    def _increment_timestep_no_fb_discretised(self, data):
        """Update the ESN state and ouputs according to the discretised equation described in [3]. Does not take feedback into account."""
        new_x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u)) 
        # print(f"Wv: {self.Wv.shape},\nx: {self.x.shape}")
        self.v = self.func(numpy.dot(self.Wv, self.x))  
        self.x = new_x
        self.u = data
        self.t += 1
        return

    def _increment_timestep_fb_discretised(self, data):
        """Update the ESN state and outputs according to the discretised equation described in [3]. Takes feedback into account."""
        # incorporates feedback of the output
        new_x = self.func(numpy.dot(self.W, self.x) + numpy.dot(self.Wu, self.u) + numpy.dot(self.Wback, self.v))
        self.v = self.func(numpy.dot(self.x, self.Wv))
        self.x = new_x
        self.u = data
        self.t += 1
        return

    def pretty_print(self):
        """Pretty print the reservoir."""
        g = igraph.Graph.Weighted_Adjacency(self.W, loops=False)
        if self.topology == "ring":
            layout = g.layout_circle()
        elif self.topology == "lattice":
            layout = g.layout_grid()
        elif self.topology == "torus":
            layout = g.layout_grid()
        else: 
            layout = g.layout_kamada_kawai()
        igraph.plot(g, layout=layout)
        return

    def train_pseudoinverse(self, training_set):
        """
        Train the Output Weight matrix using the pseudo-inverse method descirbed in [4].

        training_set: Dict of the training set, of form {input: target output}.
        
        Note: [4] describes this method as "it is expensive memory-wise for large design matrices [self.x]". That being said, given the scale of this ESN, it is perhaps the most straightforward option, as it is very simple to understand.
        """
        # make sure that the size of th

        if list(training_set.values())[0].size != self.v.size:
            raise ValueError(f"please make sure your ESN outputs match your training set: {list(training_set.values())[0].size}, {self.v.size}")
        for data in training_set:
            self._increment_timestep(data)
            if self.t <= 10:
                pass
            elif self.M_train is None:
                self.M_train = self.x # note: here i am going by the assumption that numpy is doign assign by value, not by reference, which as far as I can tell is true
                # print(f"M (pre-reshape): {self.M_train.shape}")
                self.M_train = numpy.transpose(self.M_train)
                self.D_train = numpy.ndarray(shape=self.v.shape, buffer=training_set[data]).T
                self.outputs = self.v
                # print(f"M = {self.M_train.shape},\n D={self.D_train.shape},\n M+={numpy.linalg.pinv(self.M_train).shape}")
                self.Wv = numpy.transpose(numpy.dot(numpy.linalg.pinv(self.M_train), self.D_train))
            else:
                # print(self.M_train.size)
                y = self.x
                y = numpy.transpose(y)
                self.M_train = numpy.vstack((self.M_train, y))
                self.D_train = numpy.vstack((self.D_train, training_set[data]))
                self.outputs = numpy.vstack((self.outputs, self.v))
                # print(f"M = {self.M_train.shape},\n D={self.D_train.shape},\n M+={numpy.linalg.pinv(self.M_train).shape}")
                # this is broken right here
                self.Wv = numpy.transpose(numpy.dot(numpy.linalg.pinv(self.M_train), self.D_train)) #Note: the output matrix derived with this method gives a transposition of the weight matrix described in [1], hence the transposition here
        return

    def test(self, testing_set):
        """
        Test the ESN.

        testing_set: Dict of testing values of form {input: target_output}.
        """
        # print(self.Wv)
        self._increment_timestep(list(testing_set.keys())[0])
        self.outputs = self.v
        for data in list(testing_set.keys())[1:]:
            self._increment_timestep(data)
            self.outputs = numpy.vstack((self.outputs, self.v))
        return 



"""
Bibliography:

[1] Jaeger, Herbert. n.d. “The ‘echo State’ Approach to Analysing and Training Recurrent Neural Networks – with an Erratum note1.” Accessed October 5, 2021. http://www.faculty.jacobs-university.de/hjaeger/pubs/EchoStatesTechRep.pdf.

[2] Dale, Matthew, Simon O’Keefe, Angelika Sebald, Susan Stepney, and Martin A. Trefzer. 2021. “Reservoir Computing Quality: Connectivity and Topology.” Natural Computing 20 (2): 205–16.

[3] Stepney, Susan. n.d. “Non-Instantaneous Information Transfer in Physical Reservoir Computing.

[4] Montavon, Grégoire, Geneviève B. Orr, and Klaus-Robert Müller, eds. 2012. Neural Networks: Tricks of the Trade: Second Edition. Springer, Berlin, Heidelberg.
"""