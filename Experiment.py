import TinyESN
import NARMA10
import numpy as numpy
from sklearn.metrics import mean_squared_error

# class Experiment():
#     def __init__(self, N):
#         self.esn = TinyESN(1, N, 1, numpy.tanh)
#         self.inputs = Dict()

    
#     def train_no_feedback(self):
#         for row in self.inputs:
#             self.esn.x = row
#             self.esn.increment_timestep_no_fb_discretised()

#     def train_feedback(self):
#         for row in self.inputs:
#             self.esn.x = row
#             self.esn.increment_timestep_fb_discretised()
        
#     def test_feedback(self):

#     def test_no_feedback(self):

narma = NARMA10.Narma10()
data = narma.create_training_set(1000)
training_set = dict(list(data.items())[len(data)//2:])
testing_set = dict(list(data.items())[:len(data)//2])
esn_pseudoinverse = TinyESN.TinyESN(1, 10, 1, numpy.tanh)
esn_pseudoinverse.train_pseudoinverse(training_set)
training_mse = mean_squared_error(list(training_set.values())[10:], esn_pseudoinverse.outputs, squared=False)
esn_pseudoinverse.test(testing_set)
testing_mse = mean_squared_error(list(testing_set.values()), esn_pseudoinverse.outputs, squared=False)
print(f"training set: {training_mse}\ntesting set: {testing_mse}")

#TODO (this afternoon, if i can be arsed: figure out how I want to test this)