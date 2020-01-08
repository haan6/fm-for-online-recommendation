
import logging
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Module

import os
import sys
sys.path.append('..')


from models.models_online.FM_Base import FM_Base
from utils.data_manager  import *


import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

Tensor_type = torch.DoubleTensor
numpy_type = np.float64


# Factorization Machines + FTRL
class FM_FTRL(FM_Base):

    # Follow-the-regularized-leader to Factorization Machine implementation
    def __init__(self, inputs_matrix, outputs, task, learning_rate, num_feature):

        super(FM_FTRL, self).__init__(inputs_matrix, outputs, task, learning_rate, num_feature)

        self.model_name = "FM_FTRL"
        #self._init_parameter()


    def _init_parameter(self):
        #self.w1 = Variable(torch.randn(self.num_feature,1).type(Tensor_type))
        #self.W2 = Variable(torch.randn(2*self.m,self.num_feature-1).type(Tensor_type) , requires_grad = True)

        self.w1 = torch.randn(self.num_feature,1).type(Tensor_type)
        self.W2 = torch.randn(2*self.m,self.num_feature-1).type(Tensor_type)



    def online_learning(self):
        start = time.time()

        self._init_parameter()

        print(self.model_name + '_' +  str(self.eta) +  '_'+ str(self.m)  + '_start')
        #print("==" * 10)
        pred_list = []
        real_list = []
        g_w1 = torch.zeros_like(self.w1)
        g_W2 = torch.zeros_like(self.W2)


        for idx in range(self.num_data):
            alpha = self.At[:, idx].unsqueeze(1)
            temp_scalar = self.W2.matmul(alpha[:-1])
            scalar,pred = self._predict( self.w1.t().matmul(alpha) + temp_scalar.t().matmul(temp_scalar) )
            if torch.isnan(scalar):
                raise ValueError('Nan contained')


            if self.task == 'cls':
                sign_idx = self._grad_loss(scalar * self.b[idx]) * self.b[idx]
            elif self.task == 'reg':
                sign_idx = self._grad_loss(scalar - self.b[idx])
            else:
                raise NotImplementedError

            # sum of current_gradient w.r.t w1,W2
            g_w1 += sign_idx*alpha
            g_W2 += 2*self.W2.matmul(alpha[:-1]).matmul(alpha[:-1].t())

            self.w1 = -self.eta*g_w1
            self.W2 = -self.eta*g_W2


            pred_list.append(pred)
            real_list.append(self.b[idx])
            if idx % 1000 == 0:
                print(' %d th : pred %f , real %f ' % (idx, pred, self.b[idx], ))


        end = time.time()
        print('learning time : %f ' % (end-start))

        return np.asarray(pred_list),np.asarray(real_list), (end - start)


if __name__ == "__main__":

    nbUsers = 943
    nbMovies = 1682
    nbFeatures = nbUsers + nbMovies
    nbRatingsTrain = 90570
    nbRatingsTest = 9430

    data_dir = './../../dataset/ml-100k/'
    print(data_dir)


    #filename1, filename2 = 'ub.base', 'ub.test'
    filename1, filename2 = 'ua.base', 'ua.test'

    print(data_dir + filename1)
    # load dataset
    _, x_train, y_train, rate_train, timestamp_train = load_dataset_movielens(data_dir + filename1,
                                                                              nbRatingsTrain,
                                                                              nbFeatures,
                                                                              nbUsers)
    # sort dataset in time
    #x_train_s, rate_train_s, _ = sort_dataset(x_train, rate_train, timestamp_train)
    x_train_s, rate_train_s, _ = sort_dataset_movielens(x_train, rate_train, timestamp_train)

    want_permute = True
    if want_permute :
        idx = np.random.permutation(x_train_s.shape[0])
        x_train_s = x_train_s[idx]
        rate_train_s = rate_train_s[idx]
    else:
        pass



    #model setup
    m = 40
    lr_FM = 0.005
    task = 'reg'


    #task, learning_rate, feature_m
    Model_FM_FTRL = FM_FTRL(Tensor_type(x_train_s.todense()),Tensor_type(rate_train_s) , task,lr_FM, m )
    pred_F, _ , time = Model_FM_FTRL.online_learning()

    print(Model_FM_FTRL.W2)





