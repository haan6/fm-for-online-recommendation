# -*- coding:utf-8 -*-

"""
@author: Yeji Han

A PyTorch Implementation of Online NFM with Hedge Backpropagation
"""

import numpy as np
from time import time

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import torch.backends.cudnn


class ONN_NFM_V4(torch.nn.Module):
    def __init__(self,
                 field_size,
                 feature_sizes,
                 max_num_hidden_layers,
                 qtd_neuron_per_hidden_layer,
                 dropout_shallow=[0.5],
                 embedding_size=4,
                 n_classes=1,
                 batch_size=1,
                 verbose=False,
                 interaction_type=True,
                 eval_metric=roc_auc_score,
                 b=0.99,
                 n=0.01,
                 s=0.2,
                 use_cuda=True,):
        super(ONN_NFM_V4, self).__init__()

        # Check CUDA
        if torch.cuda.is_available() and use_cuda:
            print("Using CUDA")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.dropout_shallow = dropout_shallow
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.verbose = verbose
        self.interaction_type = interaction_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda

        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)

        # FM Part
        self.first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1)
                                                     for feature_size in self.feature_sizes]).to(self.device)
        if self.dropout_shallow:
            self.first_order_dropout = nn.Dropout(self.dropout_shallow[0]).to(self.device)
        self.second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size)
                                                      for feature_size in self.feature_sizes]).to(self.device)
        self.bias = Parameter(torch.randn(1)).to(self.device)

        # Neural Networks Part

        self.hidden_layers = []
        self.output_layers = []

        if self.interaction_type:
            self.hidden_layers.append(nn.Linear(embedding_size, qtd_neuron_per_hidden_layer))
        else:
            self.hidden_layers.append(
                nn.Linear(self.field_size * (self.field_size - 1) / 2, qtd_neuron_per_hidden_layer))

        for i in range(max_num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))

        for i in range(max_num_hidden_layers):
            self.output_layers.append(nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)

        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers).fill_(1 / (self.max_num_hidden_layers + 1)),
                               requires_grad=False).to(self.device)

        self.loss_array = []

    def zero_grad(self):
        for i in range(self.max_num_hidden_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)



    def first_order(self, Xi, Xv):
        Xi = torch.LongTensor(Xi).to(self.device).reshape(-1, self.field_size, 1)
        Xv = torch.FloatTensor(Xv).to(self.device).reshape(-1, self.field_size)

        first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
                               for i, emb in enumerate(self.first_order_embeddings)]
        first_order = torch.cat(first_order_emb_arr, 1)

        return first_order



    def second_order(self, Xi, Xv):
        # FM Part
        Xi = torch.LongTensor(Xi).to(self.device).reshape(-1, self.field_size, 1)
        Xv = torch.FloatTensor(Xv).to(self.device).reshape(-1, self.field_size)


        # print('In second_order Xi')
        # print(Xi)
        # print(Xi.shape)
        second_order_emb_arr = [torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i].t()
                                for i, emb in enumerate(self.second_order_embeddings)]

        # second_order_emb_arr_v2 = []
        # for i, emb in enumerate(self.second_order_embeddings):
        #     second_order_emb_arr_v2.append(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i].t())
        # # print(second_order_emb_arr)
        # # print(second_order_emb_arr_v2)
        # for i_th,i_th_arr in enumerate(second_order_emb_arr):
        #     print(i_th)
        #     print((i_th_arr-second_order_emb_arr_v2[i_th]))
        #     print('')

        sum_second_order_emb = sum(second_order_emb_arr)
        #print(sum_second_order_emb)

        # (xi+xj)^2
        sum_second_order_emb_square = sum_second_order_emb * sum_second_order_emb
        # xi^2+xj^2
        second_order_emb_square = [item * item for item in second_order_emb_arr]
        second_order_emb_square_sum = sum(second_order_emb_square)
        second_order = (sum_second_order_emb_square - second_order_emb_square_sum) * 0.5

        return second_order.t()


    def forward_fm(self,Xi,Xv):
        first_order = self.first_order(Xi, Xv)
        second_order = self.second_order(Xi, Xv)
        total_sum = self.b + torch.sum(first_order, 1) + torch.sum(second_order, 1)
        return total_sum


    def forward_onn(self, Xi, Xv):
        # Neural Networks P
        x = self.second_order(Xi, Xv)

        hidden_connections = []
        x = F.relu(self.hidden_layers[0](x))
        hidden_connections.append(x)

        for i in range(1, self.max_num_hidden_layers):
            hidden_connections.append(F.relu(self.hidden_layers[i](hidden_connections[i - 1])))

        output_class = []
        for i in range(self.max_num_hidden_layers):
            output_class.append(self.output_layers[i](hidden_connections[i]))

        pred_per_layer = torch.stack(output_class)
        return pred_per_layer


    # def forward(self, Xi, Xv):
    #
    #     # Neural Networks P
    #     x = self.second_order(Xi, Xv)
    #
    #     hidden_connections = []
    #     x = F.relu(self.hidden_layers[0](x))
    #     hidden_connections.append(x)
    #
    #     for i in range(1, self.max_num_hidden_layers):
    #         hidden_connections.append(F.relu(self.hidden_layers[i](hidden_connections[i - 1])))
    #
    #     output_class = []
    #     for i in range(self.max_num_hidden_layers):
    #         output_class.append(self.output_layers[i](hidden_connections[i]))
    #
    #     pred_per_layer = torch.stack(output_class)
    #     return pred_per_layer


    def update_embedding(self,Xi, Xv, Y):
        # Xi = Variable(torch.LongTensor(Xi).reshape(-1, self.field_size, 1)).to(self.device)
        # Xv = Variable(torch.FloatTensor(Xv).reshape(-1, self.field_size)).to(self.device)
        Y = Variable(torch.FloatTensor(Y)).to(self.device)

        self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.n)
        criterion = F.binary_cross_entropy_with_logits

        optimizer.zero_grad()
        output = self.forward_fm(Xi, Xv)


        loss = criterion( torch.sigmoid(output) , Y)
        loss.backward()
        optimizer.step()
        return



    def update_weights(self, Xi, Xv, Y, show_loss):
        Y = torch.from_numpy(Y).to(self.device)
        predictions_per_layer = self.forward_onn(Xi, Xv)

        # print('update weight in')
        # print(predictions_per_layer)

        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.BCELoss().to(self.device)
            loss = criterion(torch.sigmoid(out.view(self.batch_size)), Y.view(self.batch_size).float() )
            losses_per_layer.append(loss)

            #print('#'*100)

        w = []
        b = []

        for i in range(len(losses_per_layer)):
            losses_per_layer[i].backward(retain_graph=True)
            self.output_layers[i].weight.data -= self.n * self.alpha[i] * self.output_layers[i].weight.grad.data
            self.output_layers[i].bias.data -= self.n * self.alpha[i] * self.output_layers[i].bias.grad.data
            w.append(self.alpha[i] * self.hidden_layers[i].weight.grad.data)
            b.append(self.alpha[i] * self.hidden_layers[i].bias.grad.data)
            self.zero_grad()

        for i in range(1, len(losses_per_layer)):
            self.hidden_layers[i].weight.data -= self.n * torch.sum(torch.cat(w[i:]))
            self.hidden_layers[i].bias.data -= self.n * torch.sum(torch.cat(b[i:]))

        for i in range(len(losses_per_layer)):
            self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
            self.alpha[i] = torch.max(self.alpha[i], self.s / self.max_num_hidden_layers)

        z_t = torch.sum(self.alpha)

        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        if show_loss:
            real_output = torch.sum(torch.mul(
                self.alpha.view(self.max_num_hidden_layers, 1).repeat(1, self.batch_size).view(
                    self.max_num_hidden_layers, self.batch_size, 1), predictions_per_layer), 0)
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(real_output.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
            self.loss_array.append(loss)
            if (len(self.loss_array) % 1000) == 0:
                print("WARNING: Set 'show_loss' to 'False' when not debugging. "
                      "It will deteriorate the fitting performance.")
                loss = torch.Tensor(self.loss_array).mean().cpu().numpy()
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))
                self.loss_array.clear()



    def partial_fit_(self, Xi_data, Xv_data, Y_data, Pseudo_Y_data, show_loss=False):

        self.update_weights(Xi_data, Xv_data, Y_data, show_loss)


        if Y_data == Pseudo_Y_data:
            # print(Y_data)
            # print(Pseudo_Y_data)
            self.update_embedding(Xi_data, Xv_data, Pseudo_Y_data)
        else :
            #pass
            self.update_embedding(Xi_data, Xv_data, Y_data)

    def partial_fit(self, Xi_data, Xv_data, Y_data, Pseudo_Y_data, show_loss=False):
        self.partial_fit_(Xi_data, Xv_data, Y_data, Pseudo_Y_data, show_loss)



    def predict_(self, Xi_data, Xv_data):
        # output = self.forward_fm(Xi_data, Xv_data)
        # output += ((self.alpha.reshape([-1, 1, 1])).mul(self.forward_onn(Xi_data, Xv_data)).sum(dim=0)).squeeze
        # ()
        output = ((self.alpha.reshape([-1, 1, 1])).mul(self.forward_onn(Xi_data, Xv_data)).sum(dim=0)).squeeze()
        pred = torch.sigmoid(output).cpu()
        return 1.0 if pred.data.numpy() > 0.5 else 0.0



    def predict(self, Xi_data, Xv_data):
        pred = self.predict_(Xi_data, Xv_data)
        return pred


    def evaluate(self, train_Xi, train_Xv, train_Y):
        accuracy = []
        roc = []
        confusion_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        start = time()
        for i in range(len(train_Y)):
            pred = self.predict(np.array(train_Xi[i]), np.array(train_Xv[i]))
            self.partial_fit(np.array(train_Xi[i]), np.array(train_Xv[i]), np.array([train_Y[i]]), np.array([pred]))

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

            if i % 1000 == 0:
                tpr = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'] + 1e-16)
                fpr = confusion_matrix['fp'] / (confusion_matrix['fp'] + confusion_matrix['tn'] + 1e-16)
                roc.append({'tpr': tpr, 'fpr': fpr})
                accuracy.append(((confusion_matrix['tp'] + confusion_matrix['tn']) / (i + 1) * 100))

        time_elapsed = time() - start
        return time_elapsed, accuracy, roc