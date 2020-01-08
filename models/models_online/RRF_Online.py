import os
import sys
import time
import torch
from torch.nn import Module
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

from utils.data_manager  import *

Tensor_type = torch.DoubleTensor
numpy_type = np.float64


class RRF_Online(Module):

    def __init__(self,inputs_matrix,
                      outputs,
                      task,
                      loss_type=None,
                      gamma=None,
                      w=None,
                      num_sampled_spectral=10,
                      random_seed=100,
                      lr_RRF_w=0.05,
                      lr_RRF_gamma=0.05):

        super(RRF_Online, self).__init__()
        self.X = inputs_matrix  # input_matrix : num_data x num_feature
        self.Y = outputs  # output : num_data x 1
        self.loss_type = loss_type
        self.num_feature = inputs_matrix.shape[1]
        self.model_name = "RRF_Online"
        self.task = task
        self.num_sampled_spectral = num_sampled_spectral
        self.lr_RRF_w = lr_RRF_w
        self.lr_RRF_gamma = lr_RRF_gamma

        self.random_seed = random_seed
        self._init_param(gamma, w, loss_type)

    # param initialization
    def _init_param(self, gamma, w, loss_type):
        if self.task == 'cls':
            self.loss_type = 'logit' if loss_type == None else 'hinge'
        elif self.task == 'reg':
            self.loss_type = 'l2' if loss_type == None else 'l1'
        else:
            raise NotImplemented('wrong task assigned')
        # gamma
        if gamma == None:
            #self.gamma = Tensor_type(np.log(1.0) * np.ones((self.num_feature, 1)))  # log 형태로 save
            self.gamma = Tensor_type(np.log( np.random.rand(self.num_feature, 1)) )  # log 형태로 save

        else :
            self.gamma = Tensor_type(np.log(gamma) * np.ones((self.num_feature, 1)))  # log 형태로 save


        if w is None:
            self.w = 0.1 * torch.randn(2 * self.num_sampled_spectral).type(Tensor_type)
        else:
            self.w = w
        # self.eps = torch.randn( self.num_feature, 2*self.num_sampled_spectral ).type(Tensor_type)
        self.eps = torch.randn(self.num_feature, self.num_sampled_spectral).type(Tensor_type)
        return

    def _compute_phi(self, x_t):
        # x_t : 1 x num_feature

        phi_t = self.gamma.exp().mul(self.eps)
        phi_x = x_t.matmul(phi_t)
        return torch.cat([phi_x.cos(), phi_x.sin()], 1)

    def _get_gradient_wrt_gamma(self, x_t, phi, d_phi):
        # def _get_gradient_wrt_gamma(self,x_t, phi):
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        d_po = torch.zeros([x_t.shape[0], 2 * self.num_sampled_spectral, self.num_feature]).type(Tensor_type)
        cos_wx, sin_wx = phi[:, :self.num_sampled_spectral], phi[:, self.num_sampled_spectral:]

        d_po[:, :self.num_sampled_spectral, :] = torch.einsum("mn,md,nd -> mdn", [-x_t, sin_wx, self.eps])
        d_po[:, self.num_sampled_spectral:, :] = torch.einsum("mn,md,nd -> mdn", [x_t, cos_wx, self.eps])
        d_gamma = torch.einsum("mdn,md->n", [d_po, d_phi]).unsqueeze(1) * self.gamma.exp()
        return d_gamma

    def _get_gradient_wrt_all(self, x_t, y_t):
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        num_batch = x_t.shape[0]
        phi = self._compute_phi(x_t)
        w_x = phi.matmul(self.w)
        d_w = self.lr_RRF_w * self.w.exp()

        #d_gamma = torch.zeros(self.gamma.shape).type(Tensor_type)
        if self.loss_type == 'logit':
            w_x_y = w_x * y_t
            c = (-y_t * (-w_x_y - torch.logsumexp(-w_x_y, 0)).exp()).unsqueeze(1)
            d_w += c.mul(phi).mean(0)
            d_phi = c.mul(self.w)
            d_gamma = 1 / num_batch * self._get_gradient_wrt_gamma(x_t, phi, d_phi)
            return d_w, d_gamma

        elif self.loss_type == 'l2':
            w_x_y = (w_x - y_t).unsqueeze(-1)
            d_w += (w_x_y * phi).mean(0)
            d_phi = w_x_y.mul(self.w)
            d_gamma = 1 / num_batch * self._get_gradient_wrt_gamma(x_t, phi, d_phi)
            return d_w, d_gamma

        elif self.loss_type == 'hinge':
            raise NotImplementedError('wrong loss type in get_grad')

        elif self.loss_type == 'l1':
            raise NotImplementedError('wrong loss type in get_grad')

        else:
            raise NotImplementedError('wrong loss type in get_grad')


    # def _predict(self, phi):
    #     assert (torch.is_tensor(phi))
    #     y_pred = phi.matmul(self.w)
    #     if self.task == 'cls':
    #         # option1
    #         #y_pred = 1 / (1 + torch.exp(-y_pred))
    #
    #         # option2
    #         #y_pred = Tensor_type( [1.0 if y_pred > 0 else -1.0 ] )
    #         y_pred = Tensor_type( [1.0 if y_pred > 0 else 0.0 ] )
    #
    #     return y_pred

    def _predict(self, phi):
        assert (torch.is_tensor(phi))
        y_pred = phi.matmul(self.w)

        return y_pred

    def online_learning(self):
        start = time.time()
        #
        print("==" * 20)
        #print(self.model_name)

        # print(self.model_name + '_' + str(self.gamma.exp()) + '_' + str(self.lr_RRF_gamma) + '_' + str(self.lr_RRF_w) + '_' + str(
        #     self.num_sampled_spectral) + '_start')

        pred_list = []
        real_list = []

        for t in range(self.X.shape[0]):
            # x_t = self.At[:, t].unsqueeze(1)
            x_t = self.X[t, :].unsqueeze(0)
            y_t = self.Y[t]
            phi_t = self._compute_phi(x_t)
            scalar = self._predict(phi_t)

            #print(scalar)
            if torch.isnan(scalar):
                #raise ValueError('Nan contained')
                #print('Nan contained')
                pass
            else:

                d_w, d_gamma = self._get_gradient_wrt_all(x_t, y_t)
                self.w -= self.lr_RRF_w * d_w
                self.gamma -= self.lr_RRF_gamma * d_gamma

                if self.task == 'cls':
                    scalar = Tensor_type( [1.0 if scalar >= 0 else -1.0 ] )

                pred_list.append(scalar)
                real_list.append(y_t)

            if t % 1000 == 0:
                # print(' %d th : pred %f , real %f , loss %f ' % (
                # idx, scalar, self.b[idx], self._loss(scalar - self.b[idx])))

                print(' %d th : pred %f , real %f ' % (t, scalar, self.Y[t],))

        end = time.time()
        print('learning time : %f ' % (end - start))

        return np.asarray(pred_list),np.asarray(real_list), (end - start)


if __name__ == "__main__":
    data_dir = './../../dataset/ml-100k/'
    filename1, filename2 = 'ua.base', 'ua.test'
    nbUsers = 943
    nbMovies = 1682
    nbFeatures = nbUsers + nbMovies
    nbRatingsTrain = 90570
    nbRatingsTest = 9430

    # load dataset
    _, x_train, y_train, rate_train, timestamp_train = load_dataset_movielens(data_dir + filename1,
                                                                              nbRatingsTrain,
                                                                              nbFeatures,
                                                                              nbUsers)
    # sort dataset in time
    # x_train_s, rate_train_s, _ = sort_dataset(x_train, rate_train, timestamp_train)
    x_train_s, rate_train_s, _ = sort_dataset_movielens(x_train, rate_train, timestamp_train)
    inputs_matrix, outputs = Tensor_type(x_train_s.todense()), Tensor_type(rate_train_s)


    Online_RRF = RRF_Online(inputs_matrix,
                            outputs,
                            'reg')

    pred_RRF = Online_RRF.online_learning()

    #print(type(x_train_s))
