import TinyESN

class Experiment():
    def __init__(self, inputs, K, N, L, f):
        self.esn = TinyESN(K, N, L, f)
        self.inputs = inputs
    
    def run_no_feedback(self):
        for row in self.inputs:
            self.esn.x = row
            self.esn.increment_timestep_no_fb_discretised()

    def run_feedback(self):
        for row in self.inputs:
            self.esn.x = row
            self.esn.increment_timestep_fb_discretised()