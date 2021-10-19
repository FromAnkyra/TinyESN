"""tests NARMA10 on the TinyESN"""
import numpy

class Narma10:
    def __init__(self, mode="discretised"):
        self.y = 0
        self.timestep = 0
        self.history = {}
        self.input = 0
        possible_modes = ["discretised", "instantaneous"]
        if mode not in possible_modes:
            raise ValueError("mode type is invalid.")
        self.mode = mode
        return
    
    def increment_timestep(self, current_input):
        if self.mode == "discretised":
            self.increment_timestep_discretised(current_input)
        else:
            self.increment_timestep_inst(current_input)
        return 
    
    def increment_timestep_discretised(self, current_input):
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

        self.y = 0.3*self.y + (0.05*self.y)*history_sum + 1.5*input_nine*self.input + 0.1
        
        self.input = current_input
        self.history[self.input] = float(self.y)
        self.timestep += 1

        return 

    def increment_timestep_inst(self, current_input):
        history_sum = 0
        r = 0
        self.input = current_input
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

        self.y = 0.3*self.y + (0.05*self.y)*history_sum + 1.5*input_nine*self.input + 0.1

        self.history[self.input] = float(self.y)
        self.timestep += 1

    
    def generate_input(self):
        return numpy.random.random_sample()/2
    
    def create_training_set_inst(self, size):
        for i in range(size):
            self.increment_timestep(self.generate_input())
        
        return self.history
    
    
