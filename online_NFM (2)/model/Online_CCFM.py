import numpy as np

import torch
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import os
import sys

from torch.nn import Module
from FM_Base import FM_Base

Tensor_type = torch.DoubleTensor


class SFTRL_CCFM(FM_Base):

    def __init__(self, inputs_matrix, outputs, task, learning_rate, num_feature):
        super(SFTRL_CCFM, self).__init__(inputs_matrix, outputs, task, learning_rate, num_feature)

        self.model_name = "SFTRL_CCFM"
        self.row_count_p = 0
        self.row_count_n = 0
        self.BT_P = Tensor_type(np.zeros([self.num_feature, 2 * self.m]))
        self.BT_N = Tensor_type(np.zeros([self.num_feature, 2 * self.m]))

    def online_learning(self, logger):
        start = time.time()
        logger.info("==" * 20)
        logger.info(self.model_name + '_' + str(self.eta) + '_' + str(self.m) + '_start')

        pred_list = []
        real_list = []
        # num_data = inputs.shape[0]
        for idx in range(self.num_data):

            alpha = self.At[:, idx].unsqueeze(1)
            # alpha = inputs_data[:, idx].unsqueeze(1)
            BP_alpha = self.BT_P.t().matmul(alpha)
            BN_alpha = self.BT_N.t().matmul(alpha)
            scalar, pred = self._predict((BP_alpha.t().matmul(BP_alpha) - BN_alpha.t().matmul(BN_alpha)).squeeze())

            if torch.isnan(scalar):
                logger.info('Nan contained')
                raise ValueError('Nan contained')

            if self.task == 'cls':
                # sign_idx = self._grad_loss(scalar * self.b[idx])*(self.b[idx])
                sign_idx = self._grad_loss(scalar * self.b[idx]).mul(self.b[idx])
            elif self.task == 'reg':
                sign_idx = self._grad_loss(scalar - self.b[idx])
            else:
                raise NotImplementedError

            self._GFD(sign_idx, alpha)

            pred_list.append(pred)
            real_list.append(self.b[idx])
            if idx % 1000 == 0:
                if self.task == 'reg':
                    c_metric = (pred - self.b[idx]) ** 2
                else:
                    c_metric = 1 if pred * self.b[idx] > 0 else 0

                logger.info(' %d th : pred %.4f,\t real %.4f,\t error %.4f ' % (idx, pred, self.b[idx], c_metric))
                print(' %d th : pred %.4f,\t real %.4f,\t error %.4f ' % (idx, pred, self.b[idx], c_metric))

        # return torch.tensor(pred_list).reshape([-1,1]).double(),\
        #        torch.tensor(np.asarray(real_list).reshape[-1,1]).double()
        end = time.time()
        logger.info('learning time : %f ' % (end - start))

        return np.asarray(pred_list), np.asarray(real_list), (end - start)

    def _GFD(self, sign, alpha):
        # alpha : num_feature x 1
        # sign : scalar value

        if sign <= 0:
            self.row_count_p += 1
            self.BT_P[:, self.row_count_p] = np.sqrt(-self.eta * sign) * alpha.squeeze()

            if self.row_count_p == 2 * self.m - 1:
                U, Sigma, _ = (self.BT_P.t().matmul(self.BT_P)).svd()
                Sigma[Sigma.data <= self._thres] = 0.0
                nnz = Sigma.nonzero().numel()
                V = self.BT_P.matmul(U[:, :nnz]).matmul((1 / Sigma[:nnz].sqrt()).diag())

                if nnz >= self.m:
                    self.BT_P = V[:, :self.m - 1].matmul((Sigma[:self.m - 1] - Sigma[self.m]).sqrt().diag())
                    self.BT_P = torch.cat([self.BT_P, torch.zeros([self.num_feature, self.m + 1]).double()], 1)
                    self.row_count_p = self.m - 1

                else:
                    self.BT_P = V[:, :nnz].matmul((Sigma[:nnz]).sqrt().diag())
                    self.BT_P = torch.cat([self.BT_P, torch.zeros([self.num_feature, (2 * self.m) - nnz]).double()], 1)
                    self.row_count_p = nnz

        else:
            self.row_count_n += 1
            self.BT_N[:, self.row_count_n] = np.sqrt(self.eta * sign) * alpha.squeeze()

            if self.row_count_n == 2 * self.m - 1:
                U, Sigma, _ = (self.BT_N.t().matmul(self.BT_N)).svd()
                Sigma[Sigma.data <= self._thres] = 0.0
                nnz = Sigma.nonzero().numel()
                V = self.BT_N.matmul(U[:, :nnz]).matmul((1 / Sigma[:nnz].sqrt()).diag())

                if nnz >= self.m:
                    self.BT_N = V[:, :self.m - 1].matmul((Sigma[:self.m - 1] - Sigma[self.m]).sqrt().diag())
                    self.BT_N = torch.cat([self.BT_N, torch.zeros([self.num_feature, self.m + 1]).double()], 1)
                    self.row_count_n = self.m - 1

                else:
                    self.BT_N = V[:, :nnz].matmul((Sigma[:nnz]).sqrt().diag())
                    self.BT_N = torch.cat([self.BT_N, torch.zeros([self.num_feature, (2 * self.m) - nnz]).double()], 1)
                    self.row_count_n = nnz

        return


class SFTRL_CCFM_v2(FM_Base):
    def __init__(self, task, learning_rate, num_feature, num_embedding, field_size, feature_sizes):

        super(SFTRL_CCFM_v2, self).__init__(inputs_matrix=None,
                                            outputs=None,
                                            task=task,
                                            learning_rate=learning_rate,
                                            num_feature=num_feature,
                                            num_embedding=num_embedding)

        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.model_name = "SFTRLCCFMv2"
        self.correct_count = 0
        self.false_count = 0
        self.row_count_p = 0
        self.row_count_n = 0
        self.eta_positive = 0.1
        self.eta_negative = 0.9
        self.weightedLossOption = False
        # self.eta_positive = 9
        # self.eta_negative = 1

        # self.BT_P = Tensor_type(np.zeros([num_feature, 2*num_embedding]))
        # self.BT_N = Tensor_type(np.zeros([num_feature, 2*num_embedding]))

        # self.BT_P = torch.randn([num_feature, 2*num_embedding]).type(Tensor_type)
        # self.BT_N = torch.randn([num_feature, 2*num_embedding]).type(Tensor_type)

        self.BT_P = torch.randn([num_feature + 1, 2 * num_embedding]).type(Tensor_type)
        self.BT_N = torch.randn([num_feature + 1, 2 * num_embedding]).type(Tensor_type)

    def _set_weighted_loss_option(self, postive_weight, negative_weight):
        self.weightedLossOption = True
        self.eta_positive = postive_weight / (postive_weight + negative_weight)
        self.eta_negative = negative_weight / (postive_weight + negative_weight)
        return

    def _predict(self, scalar):
        if self.task == 'reg':
            return scalar, scalar
        elif self.task == 'cls':
            if scalar >= 0:
                return scalar, torch.tensor([1.0]).type(Tensor_type)
            else:
                return scalar, torch.tensor([-1.0]).type(Tensor_type)
            # return 1 / (1 + np.exp(-scalar)) - 0.5
        else:
            raise NotImplementedError
            return

    def run_batch_exp(self, train_Xi, train_Xv, train_Y):
        confusion_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        roc = []
        time_elapsed = 0
        pos_pred = 0
        neg_pred = 0

        start = time.time()
        for i in range(len(train_Y)):
            # pred = self.predict(np.array(train_Xi[i]), np.array(train_Xv[i]))
            # self.partial_fit(np.array(train_Xi[i]), np.array(train_Xv[i]), np.array(train_Y[i]))
            pred, _, _ = self.online_learning_1dat_v2(train_Xi[i], np.asarray(train_Y[i]))

            if (pred == 1):
                pos_pred += 1
            else:
                neg_pred += 1

            if pred == train_Y[i]:
                if train_Y[i] == 1.0:
                    confusion_matrix["tp"] += 1
                else:
                    confusion_matrix["tn"] += 1
            else:
                if train_Y[i] == 0.0:
                    confusion_matrix["fn"] += 1
                else:
                    confusion_matrix["fp"] += 1

        tpr = confusion_matrix['tp'] / (1e-16 + confusion_matrix['tp'] + confusion_matrix['fp'])
        fpr = confusion_matrix['fp'] / (1e-16 + confusion_matrix['fp'] + confusion_matrix['tn'])
        roc = {'tpr': tpr, 'fpr': fpr}
        time_elapsed = time.time() - start

        return roc, confusion_matrix, time_elapsed

    def _idx_to_sparsevec(self, train_Xi_i_th):
        # self.field_size = field_size
        # self.feature_sizes = feature_sizes
        temp_idx = []
        for i_th in range(self.field_size):
            if i_th == self.field_size - 1:
                # temp_idx_i_th = np.zeros(self.feature_sizes[i_th])
                # temp_idx_i_th[int(train_Xi_i_th[i_th])] = 1.0
                # temp_idx.append(temp_idx_i_th)
                temp_idx.append(
                    np.asarray(list(map(int, np.binary_repr(int(train_Xi_i_th[i_th]), width=18))), dtype=np.float64))
            else:
                temp_idx_i_th = np.zeros(self.feature_sizes[i_th])
                temp_idx_i_th[int(train_Xi_i_th[i_th])] = 1.0
                temp_idx.append(temp_idx_i_th)
        return np.concatenate(temp_idx, axis=0)

    def online_learning_1dat_v2(self, train_Xi_i_th, train_Y_i_th):
        sparse_input = self._idx_to_sparsevec(train_Xi_i_th)
        # print(sparse_input)

        sparse_inputs = Tensor_type(np.concatenate((sparse_input, 1.0), axis=None))

        train_Y_i_th = 1.0 if train_Y_i_th > 0 else -1.0
        output = Tensor_type(train_Y_i_th)

        alpha = sparse_inputs.unsqueeze(1)
        # alpha = inputs_data[:, idx].unsqueeze(1)

        BP_alpha = self.BT_P.t().matmul(alpha)
        BN_alpha = self.BT_N.t().matmul(alpha)
        scalar, pred = self._predict((BP_alpha.t().matmul(BP_alpha) - BN_alpha.t().matmul(BN_alpha)).squeeze())

        if self.task == 'cls':
            # sign_idx = self._grad_loss(scalar * self.b[idx])*(self.b[idx])
            sign_idx = self._grad_loss(scalar * output).mul(output)
        elif self.task == 'reg':
            sign_idx = self._grad_loss(scalar - output)
        else:
            raise NotImplementedError

        if self.weightedLossOption:
            self._GFD_v2(sign_idx, alpha)
        else:
            self._GFD(sign_idx, alpha)

        if self.task == 'reg':
            c_metric = (pred - output) ** 2
        else:
            c_metric = 1 if pred * output > 0 else 0
            if c_metric == 1:
                self.correct_count += 1
            else:
                self.false_count += 1

        pred = 1.0 if pred > 0 else 0.0
        return np.asarray(pred), np.asarray(output), c_metric

    def online_learning_1dat(self, sparse_input, output, logger):

        # sparse_inputs = Tensor_type(sparse_input)
        sparse_inputs = Tensor_type(np.concatenate((sparse_input, 1.0), axis=None))

        output = Tensor_type(output)

        alpha = sparse_inputs.unsqueeze(1)
        # alpha = inputs_data[:, idx].unsqueeze(1)

        BP_alpha = self.BT_P.t().matmul(alpha)
        BN_alpha = self.BT_N.t().matmul(alpha)
        scalar, pred = self._predict((BP_alpha.t().matmul(BP_alpha) - BN_alpha.t().matmul(BN_alpha)).squeeze())

        if torch.isnan(scalar):
            logger.info('Nan contained')
            raise ValueError('Nan contained')

        if self.task == 'cls':
            # sign_idx = self._grad_loss(scalar * self.b[idx])*(self.b[idx])
            sign_idx = self._grad_loss(scalar * output).mul(output)
        elif self.task == 'reg':
            sign_idx = self._grad_loss(scalar - output)
        else:
            raise NotImplementedError

        if self.weightedLossOption:
            self._GFD_v2(sign_idx, alpha)
        else:
            self._GFD(sign_idx, alpha)

        if self.task == 'reg':
            c_metric = (pred - output) ** 2
        else:
            c_metric = 1 if pred * output > 0 else 0
            if c_metric == 1:
                self.correct_count += 1
            else:
                self.false_count += 1

        return np.asarray(pred), np.asarray(output), c_metric

    def _GFD(self, sign, alpha):
        # alpha : num_feature x 1
        # sign : scalar value
        if sign <= 0:
            # print(self.row_count_p)
            if self.row_count_p < 2 * self.num_embedding - 1:
                self.row_count_p += 1
            else:
                print('row_count_p : %d' % (self.row_count_p))
                pass

            self.BT_P[:, self.row_count_p] = np.sqrt(-self.eta * sign) * alpha.squeeze()

            if self.row_count_p == 2 * self.num_embedding - 1:
                U, Sigma, _ = (self.BT_P.t().matmul(self.BT_P)).svd()
                Sigma[Sigma.data <= self._thres] = 0.0
                nnz = Sigma.nonzero().numel()
                V = self.BT_P.matmul(U[:, :nnz]).matmul((1 / Sigma[:nnz].sqrt()).diag())

                if nnz >= self.num_embedding:
                    self.BT_P = V[:, :self.num_embedding - 1].matmul(
                        (Sigma[:self.num_embedding - 1] - Sigma[self.num_embedding]).sqrt().diag())
                    # self.BT_P = torch.cat([self.BT_P, torch.zeros([self.num_feature, self.num_embedding + 1]).double() ], 1)
                    self.BT_P = torch.cat(
                        [self.BT_P, torch.zeros([self.num_feature + 1, self.num_embedding + 1]).double()], 1)
                    self.row_count_p = self.num_embedding - 1

                else:
                    self.BT_P = V[:, :nnz].matmul((Sigma[:nnz]).sqrt().diag())
                    # self.BT_P= torch.cat([self.BT_P, torch.zeros([self.num_feature, (2 * self.num_embedding) - nnz]).double() ], 1)
                    self.BT_P = torch.cat(
                        [self.BT_P, torch.zeros([self.num_feature + 1, (2 * self.num_embedding) - nnz]).double()], 1)
                    self.row_count_p = nnz
            else:
                pass

        else:
            # print(self.row_count_n)
            if self.row_count_n < 2 * self.num_embedding - 1:
                self.row_count_n += 1
            else:
                print('row_count_n : %d' % (self.row_count_n))
                pass

            self.BT_N[:, self.row_count_n] = np.sqrt(self.eta * sign) * alpha.squeeze()

            if self.row_count_n == 2 * self.num_embedding - 1:
                U, Sigma, _ = (self.BT_N.t().matmul(self.BT_N)).svd()
                Sigma[Sigma.data <= self._thres] = 0.0
                nnz = Sigma.nonzero().numel()
                V = self.BT_N.matmul(U[:, :nnz]).matmul((1 / Sigma[:nnz].sqrt()).diag())

                if nnz >= self.num_embedding:
                    self.BT_N = V[:, :self.num_embedding - 1].matmul(
                        (Sigma[:self.num_embedding - 1] - Sigma[self.num_embedding]).sqrt().diag())
                    # self.BT_N = torch.cat([self.BT_N, torch.zeros([self.num_feature, self.num_embedding + 1 ]).double() ], 1)
                    self.BT_N = torch.cat(
                        [self.BT_N, torch.zeros([self.num_feature + 1, self.num_embedding + 1]).double()], 1)
                    self.row_count_n = self.num_embedding - 1

                else:
                    self.BT_N = V[:, :nnz].matmul((Sigma[:nnz]).sqrt().diag())
                    # self.BT_N = torch.cat([self.BT_N, torch.zeros([self.num_feature, (2 * self.num_embedding) - nnz] ).double()] , 1)
                    self.BT_N = torch.cat(
                        [self.BT_N, torch.zeros([self.num_feature + 1, (2 * self.num_embedding) - nnz]).double()], 1)

                    self.row_count_n = nnz
            else:
                pass
        return

    def _GFD_v2(self, sign, alpha):
        # alpha : num_feature x 1
        # sign : scalar value
        if sign <= 0:
            # print(self.row_count_p)
            if self.row_count_p < 2 * self.num_embedding - 1:
                self.row_count_p += 1
            else:
                print('row_count_p : %d' % (self.row_count_p))
                pass

            self.BT_P[:, self.row_count_p] = np.sqrt(-self.eta_negative * self.eta * sign) * alpha.squeeze()

            if self.row_count_p == 2 * self.num_embedding - 1:
                U, Sigma, _ = (self.BT_P.t().matmul(self.BT_P)).svd()
                Sigma[Sigma.data <= self._thres] = 0.0
                nnz = Sigma.nonzero().numel()
                V = self.BT_P.matmul(U[:, :nnz]).matmul((1 / Sigma[:nnz].sqrt()).diag())

                if nnz >= self.num_embedding:
                    self.BT_P = V[:, :self.num_embedding - 1].matmul(
                        (Sigma[:self.num_embedding - 1] - Sigma[self.num_embedding]).sqrt().diag())
                    self.BT_P = torch.cat(
                        [self.BT_P, torch.zeros([self.num_feature + 1, self.num_embedding + 1]).double()], 1)
                    self.row_count_p = self.num_embedding - 1
                else:
                    self.BT_P = V[:, :nnz].matmul((Sigma[:nnz]).sqrt().diag())
                    self.BT_P = torch.cat(
                        [self.BT_P, torch.zeros([self.num_feature + 1, (2 * self.num_embedding) - nnz]).double()], 1)
                    self.row_count_p = nnz
            else:
                pass

        else:
            # print(self.row_count_n)
            if self.row_count_n < 2 * self.num_embedding - 1:
                self.row_count_n += 1
            else:
                print('row_count_n : %d' % (self.row_count_n))
                pass

            self.BT_N[:, self.row_count_n] = np.sqrt(self.eta_positive * self.eta * sign) * alpha.squeeze()

            if self.row_count_n == 2 * self.num_embedding - 1:
                U, Sigma, _ = (self.BT_N.t().matmul(self.BT_N)).svd()
                Sigma[Sigma.data <= self._thres] = 0.0
                nnz = Sigma.nonzero().numel()
                V = self.BT_N.matmul(U[:, :nnz]).matmul((1 / Sigma[:nnz].sqrt()).diag())

                if nnz >= self.num_embedding:
                    self.BT_N = V[:, :self.num_embedding - 1].matmul(
                        (Sigma[:self.num_embedding - 1] - Sigma[self.num_embedding]).sqrt().diag())
                    self.BT_N = torch.cat(
                        [self.BT_N, torch.zeros([self.num_feature + 1, self.num_embedding + 1]).double()], 1)
                    self.row_count_n = self.num_embedding - 1

                else:
                    self.BT_N = V[:, :nnz].matmul((Sigma[:nnz]).sqrt().diag())
                    self.BT_N = torch.cat(
                        [self.BT_N, torch.zeros([self.num_feature + 1, (2 * self.num_embedding) - nnz]).double()], 1)

                    self.row_count_n = nnz

            else:
                pass

        return

    def _intermediate_init(self):
        self.correct_count, self.false_count = 0, 0
        nnz_p, nnz_n = self.BT_P.shape[1], self.BT_N.shape[1]

        if nnz_p < 2 * self.num_embedding:
            self.BT_P = torch.cat(
                [self.BT_P, torch.zeros([self.num_feature + 1, (2 * self.num_embedding) - nnz_p]).double()], 1)
            # self.BT_P = torch.cat([self.BT_P,torch.zeros([self.num_feature , (2 * self.num_embedding) -  nnz_p ]).double()  ], 1 )

        if nnz_n < 2 * self.num_embedding:
            self.BT_N = torch.cat(
                [self.BT_N, torch.zeros([self.num_feature + 1, (2 * self.num_embedding) - nnz_p]).double()], 1)
            # self.BT_N = torch.cat([self.BT_N,torch.zeros([self.num_feature  , (2 * self.num_embedding) -  nnz_p ]).double()  ], 1 )

        return


if __name__ == "__main__":
    print('hi')