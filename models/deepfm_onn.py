# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time

from torch.autograd import Variable


class DeepFMOnn(nn.Module):
    def __init__(self, feature_sizes, embedding_size=4, num_hidden_layers=2, neuron_per_hidden_layer=32,
                 batch_size=1, num_classes=1, b=0.99, n=0.01, s=0.2, use_cuda=True):
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
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dtype = torch.long
        self.bias = torch.nn.Parameter(torch.rand(1)).to(self.device)

        self.b = torch.nn.Parameter(torch.tensor(b)).to(self.device)
        self.n = torch.nn.Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.s = torch.nn.Parameter(torch.tensor(s), requires_grad=False).to(self.device)

        self.first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1)
                                                     for feature_size in self.feature_sizes]).to(self.device)
        self.second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size)
                                                      for feature_size in self.feature_sizes]).to(self.device)

        self.hidden_layers = []
        self.hidden_layers.append(nn.Linear(embedding_size, neuron_per_hidden_layer))
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(neuron_per_hidden_layer, neuron_per_hidden_layer))
        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)

        # self.output_layers = []
        # for i in range(num_hidden_layers):
        #     self.output_layers.append(nn.Linear(neuron_per_hidden_layer, num_classes))
        # self.output_layers = nn.ModuleList(self.output_layers).to(self.device)

        self.alpha = torch.nn.Parameter(torch.Tensor(self.num_hidden_layers).fill_(1 / (self.num_hidden_layers + 1)),
                               requires_grad=False).to(self.device)

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
        fm_part = self.forward_fm(Xi, Xv)
        x = self.second_order(Xi, Xv)
        activation = F.relu
        pred_per_layer = []

        x = activation(self.hidden_layers[0](x))
        pred_per_layer.append(torch.sigmoid(fm_part + torch.sum(x, 1)))

        for i in range(1, self.num_hidden_layers):
            x = activation(self.hidden_layers[i](x))
            pred_per_layer.append(torch.sigmoid(fm_part + torch.sum(x, 1)))

        pred_per_layer = torch.stack(pred_per_layer)
        return pred_per_layer[-1], pred_per_layer

    def zero_grad(self):
        for i in range(self.num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def fit(self, Xi, Xv, Y):
        Y = Variable(torch.FloatTensor(Y)).to(self.device)

        model = self.train()
        output, pred_per_layer = model(Xi, Xv)

        losses_per_layer = []

        for out in pred_per_layer:
            criterion = nn.BCELoss().to(self.device)
            loss = criterion(out.view(self.batch_size), Y.view(self.batch_size).float())
            losses_per_layer.append(loss)

        w = [None] * len(losses_per_layer)
        b = [None] * len(losses_per_layer)

        with torch.no_grad():
            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph=True)
                # self.output_layers[i].weight.dataset -= self.n * \
                #                                      self.alpha[i] * self.output_layers[i].weight.grad.dataset
                # self.output_layers[i].bias.dataset -= self.n * \
                #                                    self.alpha[i] * self.output_layers[i].bias.grad.dataset

                for j in range(i + 1):
                    if w[j] is None:
                        w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                    else:
                        w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            for i in range(len(losses_per_layer)):
                self.hidden_layers[i].weight.data -= self.n * w[i]
                self.hidden_layers[i].bias.data -= self.n * b[i]

            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / self.num_hidden_layers)

            z_t = torch.sum(self.alpha)
            self.alpha = torch.nn.Parameter(
                self.alpha / z_t, requires_grad=False).to(self.device)

    def update_embedding(self,Xi, Xv, Y):
        Y = Variable(torch.FloatTensor(Y)).to(self.device)

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.n)
        criterion = F.binary_cross_entropy_with_logits

        optimizer.zero_grad()
        output = self.forward_fm(Xi, Xv)

        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, Xi, Xv):
        model = self.eval()
        output, _ = model(Xi, Xv)
        pred = torch.sigmoid(output).cpu()
        return pred.cpu().data.numpy() > 0.5

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

        time_elapsed = time() - start
        return time_elapsed, accuracy[-1], roc[-1], confusion_matrix

    def __str__(self):
        return f"DeepFMOnn-Feature_Sizes{self.feature_sizes}-Embedding_Sizes{self.embedding_size}-" \
               f"Num_Hidden_Layers{self.num_hidden_layers}-Neuron_Per_Hidden_Layer{self.neuron_per_hidden_layer}-" \
               f"Num_Classes{self.num_classes}-N{self.n}"