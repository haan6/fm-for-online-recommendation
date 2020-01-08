# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time
from torch.autograd import Variable

import numpy as np


class AFMAdam(torch.nn.Module):
    def __init__(self, feature_sizes, embedding_size=4, attention_size=4, n_epochs = 64, batch_size = 256,
                 num_classes=1, b=0.99, n=0.003, use_cuda=True):
        super(AFMAdam, self).__init__()

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.bias = torch.nn.Parameter(torch.tensor(b)).to(self.device)
        self.n = torch.nn.Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.use_cuda = use_cuda

        self.first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1)
                                                     for feature_size in self.feature_sizes]).to(self.device)
        self.second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size)
                                                      for feature_size in self.feature_sizes]).to(self.device)

        self.attention_linear = nn.Linear(self.embedding_size, self.attention_size)
        self.H = torch.nn.Parameter(torch.randn(self.attention_size))
        self.P = torch.nn.Parameter(torch.randn(self.embedding_size))

    def forward(self, Xi, Xv):
        Xi = torch.LongTensor(Xi).to(self.device).reshape(-1, self.field_size, 1)
        Xv = torch.FloatTensor(Xv).to(self.device).reshape(-1, self.field_size)

        # FM
        first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
                               for i, emb in enumerate(self.first_order_embeddings)]
        first_order = torch.cat(first_order_emb_arr, 1)

        second_order_emb_arr = [torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i].t()
                                for i, emb in enumerate(self.second_order_embeddings)]
        sum_second_order_emb = sum(second_order_emb_arr)
        # (xi+xj)^2
        sum_second_order_emb_square = sum_second_order_emb * sum_second_order_emb
        # xi^2+xj^2
        second_order_emb_square = [item * item for item in second_order_emb_arr]
        second_order_emb_square_sum = sum(second_order_emb_square)
        second_order = (sum_second_order_emb_square - second_order_emb_square_sum) * 0.5

        interaction_layer = torch.cat(second_order_emb_square, 1)
        activation = F.relu

        attention_tmp = self.attention_linear(interaction_layer.view([-1, self.embedding_size]))
        attention_tmp = attention_tmp * self.H
        attention_tmp = torch.sum(attention_tmp, 1).view([-1, self.field_size*(self.field_size-1)/2])
        attention_weight = torch.nn.Softmax()(attention_tmp)
        attention_output = torch.sum(interaction_layer.view([-1,self.embedding_size])* self.P,1).view([-1,self.field_size*(self.field_size-1)/2])
        attention_output = attention_output * attention_weight

        total_sum = self.bias + torch.sum(first_order,1) + torch.sum(attention_output,1)

        return total_sum

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None,
                y_valid = None):

        is_valid = False
        Xi_train = np.array(Xi_train).reshape((-1,self.field_size,1))
        Xv_train = np.array(Xv_train)
        y_train = np.array(y_train)
        x_size = Xi_train.shape[0]

        if Xi_valid:
            Xi_valid = np.array(Xi_valid).reshape((-1,self.field_size,1))
            Xv_valid = np.array(Xv_valid)
            y_valid = np.array(y_valid)
            x_valid_size = Xi_valid.shape[0]
            is_valid = True

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.n)

        criterion = F.binary_cross_entropy_with_logits

        train_result = []
        valid_result = []

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            batch_iter = x_size // self.batch_size
            epoch_begin_time = time()
            batch_begin_time = time()
            for i in range(batch_iter+1):
                offset = i*self.batch_size
                end = min(x_size, offset+self.batch_size)
                if offset == end:
                    break

                batch_xi = Variable(torch.LongTensor(Xi_train[offset:end])).to(self.device)
                batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end])).to(self.device)
                batch_y = Variable(torch.FloatTensor(y_train[offset:end])).to(self.device)
                optimizer.zero_grad()
                outputs = self.forward(batch_xi, batch_xv)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.data[0]
                if self.verbose:
                    if i % 100 == 99:  # print every 100 mini-batches
                        eval = self.evaluate(batch_xi, batch_xv, batch_y)
                        print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                              (epoch + 1, i + 1, total_loss/100.0, eval, time()-batch_begin_time))
                        total_loss = 0.0
                        batch_begin_time = time()

            train_loss, train_eval = self.eval_by_batch(Xi_train, Xv_train, y_train, x_size)
            train_result.append(train_eval)
            print('*'*50)
            print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                  (epoch + 1, train_loss, train_eval, time()-epoch_begin_time))
            print('*'*50)

            if is_valid:
                valid_loss, valid_eval = self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
                valid_result.append(valid_eval)
                print('*' * 50)
                print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                      (epoch + 1, valid_loss, valid_eval,time()-epoch_begin_time))
                print('*' * 50)

    def eval_by_batch(self,Xi, Xv, y, x_size):
        total_loss = 0.0
        y_pred = []
        batch_size = 16384
        batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()

        for i in range(batch_iter+1):
            offset = i * batch_size
            end = min(x_size, offset + batch_size)
            if offset == end:
                break
            batch_xi = Variable(torch.LongTensor(Xi[offset:end])).to(self.device)
            batch_xv = Variable(torch.FloatTensor(Xv[offset:end])).to(self.device)
            batch_y = Variable(torch.FloatTensor(y[offset:end])).to(self.device)

            outputs = model(batch_xi, batch_xv)
            pred = F.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.data[0]*(end-offset)

        total_metric = self.eval_metric(y, y_pred)
        return total_loss/x_size, total_metric

    def predict(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1,self.field_size,1))
        Xi = Variable(torch.LongTensor(Xi)).to(self.device)
        Xv = Variable(torch.FloatTensor(Xv)).to(self.device)

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def predict_proba(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi)).to(self.device)
        Xv = Variable(torch.FloatTensor(Xv)).to(self.device)

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()

    def predict(self, Xi, Xv):
        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy() > 0.5

    def run_experiment(self, data_Xi, data_Xv, data_Y):
        data_size = len(data_Y)
        confusion_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        accuracy = []
        roc = []

        pred = self.predict(data_Xi, data_Xv)

        start = time()
        for i in range(data_size):
            if pred[i] == data_Y[i]:
                if data_Y[i] == 1:
                    confusion_matrix["tp"] += 1
                else:
                    confusion_matrix["tn"] += 1
            else:
                if data_Y[i] == 1:
                    confusion_matrix["fn"] += 1
                else:
                    confusion_matrix["fp"] += 1

            if i % 1000 == 0 or i == data_size - 1:
                tpr = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'] + 1e-16)
                fpr = confusion_matrix['fp'] / (confusion_matrix['fp'] + confusion_matrix['tn'] + 1e-16)
                roc.append({'tpr': tpr, 'fpr': fpr})
                accuracy.append(((confusion_matrix['tp'] + confusion_matrix['tn']) / (i + 1) * 100))

        time_elapsed = time() - start
        return time_elapsed, accuracy[-1], roc[-1], confusion_matrix
