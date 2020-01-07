# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time

from torch.autograd import Variable


class NFMAdam(nn.Module):
    def __init__(self, feature_sizes, embedding_size=4, num_hidden_layers=2, neuron_per_hidden_layer=32, num_classes=1,
                 b=0.99, n=0.01, use_cuda=True):
        super().__init__()

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.neuron_per_hidden_layer = neuron_per_hidden_layer
        self.num_classes = num_classes
        self.dtype = torch.long
        self.bias = torch.nn.Parameter(torch.tensor(b)).to(self.device)
        self.n = torch.nn.Parameter(torch.tensor(n), requires_grad=False).to(self.device)

        self.first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1)
                                                     for feature_size in self.feature_sizes]).to(self.device)
        self.second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size)
                                                      for feature_size in self.feature_sizes]).to(self.device)

        self.hidden_layers = []
        self.hidden_layers.append(nn.Linear(embedding_size, neuron_per_hidden_layer))

        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(neuron_per_hidden_layer, neuron_per_hidden_layer))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)

    def first_order(self, Xi, Xv):
        Xi = torch.LongTensor(Xi).to(self.device).reshape(-1, self.field_size, 1)
        Xv = torch.FloatTensor(Xv).to(self.device).reshape(-1, self.field_size)

        first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
                               for i, emb in enumerate(self.first_order_embeddings)]
        first_order = torch.cat(first_order_emb_arr, 1)

        return first_order

    def second_order(self, Xi, Xv):
        Xi = torch.LongTensor(Xi).to(self.device).reshape(-1, self.field_size, 1)
        Xv = torch.FloatTensor(Xv).to(self.device).reshape(-1, self.field_size)

        second_order_emb_arr = [torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i].t()
                                for i, emb in enumerate(self.second_order_embeddings)]
        sum_second_order_emb = sum(second_order_emb_arr)
        # (xi+xj)^2
        sum_second_order_emb_square = sum_second_order_emb * sum_second_order_emb
        # xi^2+xj^2
        second_order_emb_square = [item * item for item in second_order_emb_arr]
        second_order_emb_square_sum = sum(second_order_emb_square)
        second_order = (sum_second_order_emb_square - second_order_emb_square_sum) * 0.5

        return second_order.t()

    def forward_fm(self, Xi, Xv):
        first_order = self.first_order(Xi, Xv)
        second_order = self.second_order(Xi, Xv)
        total_sum = torch.sum(first_order, 1) + torch.sum(second_order, 1) + self.bias

        return total_sum

    def forward(self, Xi, Xv):
        fm_first = torch.sum(self.first_order(Xi, Xv), 1) + self.bias
        x = self.second_order(Xi, Xv)
        activation = F.relu
        x = activation(self.hidden_layers[0](x))

        for i in range(1, self.num_hidden_layers):
            x = activation(self.hidden_layers[i](x))

        total_sum = fm_first + torch.sum(x, 1)
        return total_sum

    def update_embedding(self,Xi, Xv, Y):
        Y = Variable(torch.FloatTensor(Y)).to(self.device)

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.n)
        criterion = F.binary_cross_entropy_with_logits

        optimizer.zero_grad()
        output = self.forward_fm(Xi, Xv)

        loss = criterion(torch.sigmoid(output), Y)
        loss.backward()
        optimizer.step()
        return loss

    def fit(self, Xi, Xv, Y):
        Y = Variable(torch.FloatTensor(Y)).to(self.device)

        model = self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.n)
        criterion = F.binary_cross_entropy_with_logits

        optimizer.zero_grad()
        output = model(Xi, Xv)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

    def predict(self, Xi, Xv):
        model = self.eval()
        output = model(Xi, Xv)
        pred = torch.sigmoid(output).cpu()
        return pred.data.numpy() > 0.5

    def run_experiment(self, data_Xi, data_Xv, data_Y):
        data_size = len(data_Y)
        confusion_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        accuracy = []
        roc = []

        start = time()
        for i in range(data_size):
            pred = self.predict(data_Xi[i], data_Xv[i])
            self.fit([data_Xi[i]], [data_Xv[i]], [data_Y[i]])

            if pred == data_Y[i]:
                if data_Y[i] == 1:
                    confusion_matrix["tp"] += 1
                else:
                    confusion_matrix["tn"] += 1
            else:
                if data_Y[i] == 1:
                    confusion_matrix["fn"] += 1
                else:
                    confusion_matrix["fp"] += 1

            if i % 1000 == 0 or i == data_size-1:
                tpr = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'] + 1e-16)
                fpr = confusion_matrix['fp'] / (confusion_matrix['fp'] + confusion_matrix['tn'] + 1e-16)
                roc.append({'tpr': tpr, 'fpr': fpr})
                accuracy.append(((confusion_matrix['tp'] + confusion_matrix['tn']) / (i + 1) * 100))

                print("ROC: ")

        time_elapsed = time() - start
        return time_elapsed, accuracy[-1], roc[-1], confusion_matrix

    def __str__(self):
        return f"NFMAdam-Feature_Sizes{self.feature_sizes}-Embedding_Sizes{self.embedding_size}-" \
               f"Num_Hidden_Layers{self.num_hidden_layers}-Neuron_Per_Hidden_Layer{self.neuron_per_hidden_layer}-" \
               f"Num_Classes{self.num_classes}"
