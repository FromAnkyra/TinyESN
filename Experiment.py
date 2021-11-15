"""Some sample experiments and operations that you might want to use to test an ESN."""
import matplotlib.pyplot as plt
import numpy
import TinyESN
from benchmark import BenchMark

class Experiment():
    """Define sample experiments that might be of use when testing an ESN.""" 
    def __init__(self):
        """Initialise the Experiment class."""
        self.default_params = {"K": 1,
                                "N": 30,
                                "L": 1,
                                "function": numpy.tanh,
                                "mode": "discretised",
                                "feedback": False,
                                "topology": "random",
                                "connectivity": 0.1,
                                "input_norm": True}
    
    def nrmse(self, target_output_set, real_output_set):#any idea why this appears to converge to 1?
        """Perform the normalised root mean squared error over the target and output sets."""
        if len(target_output_set) != len(real_output_set):
            raise ValueError(f"Found input variables with inconstitent numbers of samples [{len(target_output_set)}, {len(real_output_set)}]")
        output_average = sum(target_output_set)/len(target_output_set)
        top = 0
        bottom = 0
        for i in range(1, len(target_output_set)):
            top += (target_output_set[i] - real_output_set[i])**2
            bottom += (target_output_set[i] - output_average)**2
        top = top/len(target_output_set)
        bottom = bottom/len(target_output_set)
        return float(numpy.sqrt(top/bottom))

    def show_esn_nrmse(self, params, benchmark: BenchMark):
        """
        Train a given ESN and makes a boxplot of the NRSMEs.

        params: tuple of the parametres for the ESN to train. (Default param examples can be found in self.default_params).
        """
        benchmark.reset()
        training, testing = self.run_many(params, 100, benchmark)
        data = [training, testing]
        plt.boxplot(data, showfliers=False)
        plt.xticks([1, 2], ["training", "testing"])
        return

    def compare_esn_nrmses(self, params_1, params_2, benchmark1: BenchMark, benchmark2: BenchMark, name_1="esn 1", name_2="esn 2"):
        """
        Train two ESNs and make boxplots of the NMSREs.
        
        params_1, params_2: tuple of the parametres for the ESNs to train. (Default param examples can be found in self.default_params).

        name_1, name2: name given to the ESNs (to use when labelling boxplots)
        """
        benchmark1.reset()
        benchmark2.reset()
        esn_1_training, esn_1_testing = self.run_many(params_1, 100, benchmark1)
        esn_2_training, esn_2_testing = self.run_many(params_2, 100, benchmark2)
        data = [esn_1_training, esn_1_testing, esn_2_training, esn_2_testing]
        plt.boxplot(data, showfliers=False)
        plt.xticks([1, 2, 3, 4], [name_1+" training", name_1+" testing", name_2+" training", name_2+" testing"])
        return

    def show_esn_behaviour(self, params, benchmark: BenchMark):
        """
        Plot the ESN outputs to the target NARMA10 outputs.

        params: tuple of the parametres for the ESN to train. (Default param examples can be found in self.default_params).

        benchmark: instantiation of a benchmark against which to train the NMSRE, such as NARMA10. 
        """
        _, axs = plt.subplots(2)
        benchmark.reset()
        data = benchmark.create_training_set(1000)
        esn = TinyESN.TinyESN(*params)
        training_set = dict(list(data.items())[(len(data)//2):])
        testing_set = dict(list(data.items())[:(len(data)//2)])
        esn.train_pseudoinverse(training_set)
        axs[0].set_title("training")
        axs[0].plot(list(training_set.values())[10:])
        # print(esn.outputs)
        axs[0].plot(esn.outputs)
        esn.test(testing_set)
        axs[1].set_title("testing")
        axs[1].plot(list(testing_set.values()))
        axs[1].plot(esn.outputs)
        return

    def compare_esn_behaviour(self, params_1, params_2, benchmark1: BenchMark, benchmark2: BenchMark, name_1="esn 1", name_2="esn 2"):
        """
        Plot the outputs of two different ESNs to the targert NARMA10 outputs.
        
        params_1, params_2: tuple of the parametres for the ESNs to train. (Default param examples can be found in self.default_params).

        name_1, name2: name given to the ESNs
        """
        _, axs = plt.subplots(2)
        benchmark1.reset()
        benchmark2.reset()
        data1 = benchmark1.create_training_set(1000)
        data2 = benchmark2.create_training_set(1000)
        esn_1 = TinyESN.TinyESN(*params_1)
        esn_2 = TinyESN.TinyESN(*params_2)
        training_set1 = dict(list(data1.items())[(len(data1)//2):])
        testing_set1 = dict(list(data1.items())[:(len(data1)//2)])
        training_set2 = dict(list(data2.items())[(len(data1)//2):])
        testing_set2 = dict(list(data2.items())[:(len(data1)//2)])
        esn_1.train_pseudoinverse(training_set1)
        esn_2.train_pseudoinverse(training_set2)
        axs[0].set_title("training")
        axs[0].plot(list(training_set1.values())[10:], label="target values 1")
        axs[0].plot(list(training_set2.values())[10:], label="target values 2")
        axs[0].plot(esn_1.outputs, label=name_1)
        axs[0].plot(esn_2.outputs, label=name_2)
        esn_1.test(testing_set1)
        esn_2.test(testing_set2)
        axs[1].set_title("testing")
        axs[1].plot(list(testing_set1.values()), label="target values 1")
        axs[1].plot(list(testing_set2.values()), label="target values 2")
        axs[1].plot(esn_1.outputs, label=name_1)
        axs[1].plot(esn_2.outputs, label=name_2)
        return

    def run_many(self, params, amount: int, benchmark: BenchMark):
        """
        Run an ESN with the given params a certain number of times, and return the resulting NRMSEs.

        params: tuple of the parametres for the ESN to train. (Default param examples can be found in self.default_params).

        amount: number of times the ESN will be run.

        returns the nrmses for the training and the testing sets.
        """
        training_nrmses = []
        testing_nrmses = []
        for _ in range(amount):
            esn = TinyESN.TinyESN(*params)
            benchmark.reset()
            data = benchmark.create_training_set(500)
            training_set = dict(list(data.items())[(len(data)//2):])
            testing_set = dict(list(data.items())[:(len(data)//2)])
            esn.train_pseudoinverse(training_set)
            training_nrmse = self.nrmse(list(training_set.values())[10:], esn.outputs[:])
            esn.test(testing_set)
            testing_nrmse = self.nrmse(list(testing_set.values()), esn.outputs)
            training_nrmses.append(training_nrmse)
            testing_nrmses.append(testing_nrmse)
            del esn
        return training_nrmses, testing_nrmses
