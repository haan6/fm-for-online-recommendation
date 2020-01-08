import torch
from torch.nn import Module
#from torch.autograd import Variable

import numpy as np

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

tensor_type = torch.DoubleTensor



class FM_Base(Module):

    def __init__(self, inputs_matrix, outputs, task, learning_rate, feature_m ):

        super(FM_Base, self).__init__()

        self.At = inputs_matrix.t()
        self.b = outputs
        self._thres = 1e-12

        self.num_data = inputs_matrix.shape[0]
        self.num_feature = inputs_matrix.shape[1]

        self.task = task
        self.eta = learning_rate
        self.m = feature_m



    def _loss(self, x):

        if self.task == 'reg' :
            return x**2
        elif self.task == 'cls':
            return 1 / (1 + torch.exp(x))
        else :
            return


    def _grad_loss(self, x):

        if self.task == 'reg' :
            return 2.0*x
        elif self.task == 'cls' :
            return -1.0 / (1.0 + torch.exp(x))
        else :
            return


    def _predict(self,scalar):
        if self.task == 'reg' :
            return scalar,scalar
        elif self.task == 'cls' :
            if scalar >=0:
                return scalar,torch.tensor([1.0]).type(tensor_type)
            else :
                return scalar,torch.tensor([-1.0]).type(tensor_type)
            #return 1 / (1 + np.exp(-scalar)) - 0.5
        else :
            raise NotImplementedError
            return

    def online_learning(self,logger):
        raise NotImplementedError
        return



if __name__ == "__main__" :

    inputs_matrix = torch.tensor(np.random.randn(5,10))
    outputs = torch.tensor(np.random.randn(5,1))
    options = {}
    options['m']  = 5
    options['eta'] = 1e-4
    options['task'] = 'reg'

    print(inputs_matrix)

    #
    # pred,real = Model_CCFM.online_learning()
    #
    # plt.figure()
    # plt.plot(pred)
    # plt.plot(real)
    # plt.show()