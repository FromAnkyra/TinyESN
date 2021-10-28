import numpy
import matplotlib.pyplot as plt
from Experiment import *
from TinyESN import *
from NARMA10 import *
from NARMA5 import *
from timeseries import *
from parity import *

# narma = NARMA.Narma()
# data = narma.create_training_set(1000)
# training_set = dict(list(data.items())[len(data)//2:])
# testing_set = dict(list(data.items())[:len(data)//2])
# esn = TinyESN.TinyESN(1, 30, 1)
# esn.train_pseudoinverse(training_set)
# training_mse = mean_squared_error(list(training_set.values())[10:], esn.outputs, squared=False)
# training_nmsre = nmsre(list(training_set.values())[10:], esn.outputs)
# fig, axs = plt.subplots(2)
# axs[0].plot([list(training_set.values())[11+i]- esn.outputs[i+1] for i in range(len(esn.outputs)-1)])
# # axs[0].plot(list(training_set.values())[10:])
# # axs[0].plot(esn.outputs)
# esn.test(testing_set)
# testing_nmsre = nmsre(list(testing_set.values()), esn.outputs)
# axs[1].plot([list(testing_set.values())[i]- esn.outputs[i] for i in range(len(esn.outputs))])
# # axs[1].plot(list(testing_set.values())[10:])
# # axs[1].plot(esn.outputs)
# plt.show()
# testing_mse = mean_squared_error(list(testing_set.values()), esn.outputs, squared=False)
# print(f"training set: {training_mse}\ntesting set: {testing_mse}")
# print(f"training set: {training_nmsre}\ntesting_set: {testing_nmsre}")

# esn_ring = TinyESN(1, 30, 1, topology="ring")
# esn_ring.pretty_print()

# esn_torus = TinyESN(1, 16, 1, topology="torus")
# esn_torus.pretty_print()

# esn_lattice = TinyESN(1, 16, 1, topology="lattice")
# esn_lattice.pretty_print()

e = Experiment()
# t_esn = TinyESN.TinyESN(1, 30, 1)
# f_esn = TinyESN.TinyESN(1, 30, 1, input_norm=False)
# e.show_esn_behaviour(t_esn)
# e.show_esn_behaviour(f_esn)
# plt.show()
# with_scaling = (1, 30, 1, numpy.tanh, "discretised", False, "random", 0.1, True)
# without_scaling = (1, 30, 1, numpy.tanh, "instantaneous", False, "random", 0.1, True)

bigge = (1, 16, 1, numpy.tanh, "instantaneous", False, "lattice", 0.1, True)
smol = (1, 4, 1, numpy.tanh, "discretised", False, "lattice", 0.1, True)
n10 = Narma5()
e.compare_esn_nrmses(bigge, smol, n10, "bigge", "smol")
p = Parity()
e.show_esn_behaviour(bigge, n10)
e.show_esn_behaviour(bigge, p)
plt.show()