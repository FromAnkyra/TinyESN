"""tests NARMA10 on the TinyESN"""
import numpy as numpy

class Narma10:
    def __init__(self):
        self.y = 0
        self.timestep = 0
        self.history = {}
        return
    
    def increment_timestep(self, input):
        history_sum = 0
        r = 0
        
        if len(self.history) == 0:
            r = 0
            input_nine = 0
            history_sum = 0
        elif len(self.history) < 9:
            r = len(self.history)
            input_nine = list(self.history.keys())[self.timestep - r]
            for i in range(r):
                history_sum += list(self.history.values())[(self.timestep-1)-i]
        else:
            r = 9
            input_nine = list(self.history.keys())[self.timestep - r]
            for i in range(r-1):
                history_sum += list(self.history.values())[(self.timestep-1)-i]

        self.y = 0.3*self.y + (0.05*self.y)*history_sum + 1.5*input_nine*input + 0.1

        self.history[input] = float(self.y)
        self.timestep += 1

        return 
    
    def generate_input(self):
        return numpy.random.ranf()/2
    
    def create_training_set(self, size):
        for i in range(size):
            self.increment_timestep(self.generate_input())
        
        return self.history
    
    
