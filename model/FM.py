# -*- coding: utf-8 -*-

"""
@author: Yeji Han

A PyTorch implementation of Online FM
"""

from time import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

import torch.backends.cudnn

class FM(torch.nn.Module):
    def __init__(self, field_size, feature_sizes, embedding_size=4, is_shallow_dropout=True,
                 dropout_shallow=[0.5], batch_size=256, interaction_type=True,
                 verbose=False, loss_type='logloss',
                 b=0.99, n=0.01, eval_metric=roc_auc_score, use_cuda=True):
        super(FM, self).__init__()

        # Check CUDA
        if torch.cuda.is_available() and use_cuda:
            print("FM: Using CUDA")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.batch_size = batch_size
        self.interaction_type = interaction_type
        self.n = n
        self.verbose = verbose
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda

        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)

        # FM
        self.first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1)
                                                     for feature_size in self.feature_sizes]).to(self.device)
        if self.dropout_shallow:
            self.first_order_dropout = nn.Dropout(self.dropout_shallow[0]).to(self.device)

        self.second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size)
                                                      for feature_size in self.feature_sizes]).to(self.device)

    def forward(self, Xi, Xv):
        first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
                               for i, emb in enumerate(self.first_order_embeddings)]
        first_order = torch.cat(first_order_emb_arr, 1)

        if self.dropout_shallow:
            first_order = self.first_order_dropout(first_order)

        if self.interaction_type:
            # Use 2xixj = (xi+xj)^2 - xi^2 - yj^2 to reduce calculation
            second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
                                    for i, emb in enumerate(self.second_order_embeddings)]
            sum_second_order_emb = sum(second_order_emb_arr)
            # (xi+xj)^2
            sum_second_order_emb_square = sum_second_order_emb * sum_second_order_emb
            # xi^2+xj^2
            second_order_emb_square = [item * item for item in second_order_emb_arr]
            second_order_emb_square_sum = sum(second_order_emb_square)
            second_order = (sum_second_order_emb_square - second_order_emb_square_sum) * 0.5

        else:
            second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
                                    for i, emb in enumerate(self.second_order_embeddings)]
            weights_fm = []
            for i in range(self.field_size):
                for j in range(i + 1, self.field_size):
                    weights_fm.append(second_order_emb_arr[i] * second_order_emb_arr[j])

        total_sum = self.b + torch.sum(first_order, 1) + torch.sum(second_order, 1)
        return total_sum

    def fit(self, Xi, Xv, Y):
        Xi = Variable(torch.LongTensor(Xi).reshape(-1, self.field_size, 1)).to(self.device)
        Xv = Variable(torch.FloatTensor(Xv).reshape(-1, self.field_size)).to(self.device)
        Y = Variable(torch.FloatTensor(Y)).to(self.device)

        model = self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.n)
        criterion = F.binary_cross_entropy_with_logits

        optimizer.zero_grad()
        output = model(Xi, Xv)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

    def predict(self, Xi_data, Xv_data):
        Xi = Variable(torch.LongTensor(Xi_data).reshape(-1, self.field_size, 1)).to(self.device)
        Xv = Variable(torch.FloatTensor(Xv_data).reshape(-1, self.field_size)).to(self.device)
        model = self.eval()
        output = model(Xi, Xv)
        pred = torch.sigmoid(output).cpu()

        return pred.data.numpy() > 0.5


    def evaluate(self, train_Xi, train_Xv, train_Y):
        train_size = len(train_Y)
        accuracy = []
        confusion_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        roc = []

        start = time()
        for i in range(self.batch_size, train_size):
            pred = self.predict(train_Xi[i], train_Xv[i])

            if pred == train_Y[i]:
                if train_Y[i] == 1:
                    confusion_matrix["tp"] += 1
                else:
                    confusion_matrix["tn"] += 1
            else:
                if train_Y[i] == 1:
                    confusion_matrix["fn"] += 1
                else:
                    confusion_matrix["fp"] += 1

            start = i - self.batch_size
            self.fit(train_Xi[start:i], train_Xv[start:i], train_Y[start:i])

            if i % 100 == 0:
                tpr = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'] + 1e-16)
                fpr = confusion_matrix['fp'] / (confusion_matrix['fp'] + confusion_matrix['tn'] + 1e-16)
                roc.append({'tpr': tpr, 'fpr': fpr})
                accuracy.append(((confusion_matrix['tp'] + confusion_matrix['tn']) / (i + 1) * 100))

        end = time()
        print(start, end)
        time_elapsed = end - start
        return time_elapsed, accuracy, roc