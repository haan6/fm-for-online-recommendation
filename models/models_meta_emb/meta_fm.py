import numpy as np
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




class FM_MetaEmbedding(torch.nn.Module):
    def __init__(self,
                 direct_feature_sizes,
                 direct_embedding_sizes,
                 NNet_input_dim,
                 is_shallow_dropout=True,
                 dropout_shallow=[0.5],
                 weight_decay=0.0,
                 b=0.0,
                 n=0.001):

        super(FM_MetaEmbedding, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.field_size = 3
        self.NNet_input_dim = NNet_input_dim
        self.control_rate = 0.75

        self.tau = 3
        self.tau_test = 1
        self.direct_feature_sizes = direct_feature_sizes + 1
        self.direct_embedding_sizes = direct_embedding_sizes

        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.n = n
        self.weight_decay = weight_decay
        self.random_seed = 1

        self.b = Parameter(torch.tensor(b), requires_grad=True).to(self.device)

        if self.dropout_shallow:
            self.first_order_dropout = nn.Dropout(self.dropout_shallow[0]).to(self.device)

        self.embeddings_post_type = Variable(torch.randn(self.direct_feature_sizes, self.direct_embedding_sizes),
                                             requires_grad=True).to(self.device)


        # shallow embedding
        self.meta_emb_dim_u = 500
        self.meta_emb_dim_p = 250
        self.share_dim = 500

        self.meta_embeddings_user = Variable(torch.randn(self.meta_emb_dim_u, self.direct_embedding_sizes),
                                             requires_grad=True).to(self.device)
        self.meta_embeddings_post = Variable(torch.randn(self.meta_emb_dim_p, self.direct_embedding_sizes),
                                             requires_grad=True).to(self.device)


        self.fc1_share = nn.Linear(np.sum(self.NNet_input_dim), self.share_dim).to(self.device)
        self.bn1_share = nn.BatchNorm1d(self.share_dim).to(self.device)

        self.fc2_u = nn.Linear(self.share_dim, self.meta_emb_dim_u).to(self.device)
        self.fc2_p = nn.Linear(self.share_dim, self.meta_emb_dim_p).to(self.device)

        self.relu = nn.ReLU().to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.n)
        self.lambda1 = 0.001
        self.lambda2 = 0.001


    def get_embedding(self, inputs, inputs_v2):
        field_inputs = np.asarray(inputs)
        try:
            second_order_emb_arr = self.embeddings_post_type[inputs_v2[:, 0]]
        except:
            print(field_inputs[:, 0])

        field_inputs.astype(np.float32)
        field_inputs = Variable(torch.Tensor(field_inputs)).to(self.device)

        weight_emb_share = self.fc1_share(field_inputs)
        weight_emb_share = self.relu(self.bn1_share(weight_emb_share))

        weight_emb_user = F.softmax(self.fc2_u(weight_emb_share) / self.tau)
        weight_emb_post = F.softmax(self.fc2_p(weight_emb_share) / self.tau)

        return weight_emb_user.matmul(self.meta_embeddings_user), weight_emb_post.matmul(
            self.meta_embeddings_post), second_order_emb_arr



    def forward(self, Xi, Xv):
        Xi = np.asarray(Xi)
        Xv = np.asarray(Xv)

        first_order = self.embeddings_post_type[Xv[:, 0]]
        second_order_embedding_output_list = self.get_embedding(Xi, Xv)
        second_order_emb_sum = 0
        second_order_emb_square_sum = 0

        for i_th_second_emb in second_order_embedding_output_list:
            second_order_emb_sum += i_th_second_emb
            second_order_emb_square_sum += i_th_second_emb * i_th_second_emb

        second_order = 0.5 * (second_order_emb_sum * second_order_emb_sum - second_order_emb_square_sum)
        total_sum = self.b + torch.sum(first_order, 1) + torch.sum(second_order, 1)

        return total_sum, second_order_embedding_output_list




    def fit(self, Xi, Xv, Y, CTR, CTR2):
        Y = Variable(torch.FloatTensor(Y)).to(self.device)
        model = self.train()
        self.optimizer.zero_grad()
        output, second_order_embedding_output_list = model(Xi, Xv)
        pair_dict_pos = {}
        pair_dict_neg = {}

        temp_loss0 = 0.0
        temp_loss1 = 0.0

        margin = 0.1
        if self.control_rate > 0.0:
            criterion = F.binary_cross_entropy_with_logits
            temp_loss0 = criterion(output, Y)

        if self.control_rate < 1.0:

            for i_th in range(len(Y)):
                if Y[i_th] == 1.0:
                    if Xv[i_th, 1] not in pair_dict_pos.keys():
                        pair_dict_pos[Xv[i_th, 1]] = [output[i_th]]
                    else:
                        pair_dict_pos[Xv[i_th, 1]].append(output[i_th])
                else:
                    if Xv[i_th, 1] not in pair_dict_neg.keys():
                        pair_dict_neg[Xv[i_th, 1]] = [output[i_th]]
                    else:
                        pair_dict_neg[Xv[i_th, 1]].append(output[i_th])

            # temp_loss1_list = []
            for i_th_user in pair_dict_pos.keys():
                try:
                    for i_th in range(len(pair_dict_pos[i_th_user])):
                        for j_th in range(len(pair_dict_neg[i_th_user])):
                            # temp_loss += -torch.log( torch.sigmoid( pair_dict_pos[i_th_user][i_th] - pair_dict_neg[i_th_user][j_th] ) + margin )
                            # temp_loss1_list.append(torch.sigmoid( pair_dict_pos[i_th_user][i_th] - pair_dict_neg[i_th_user][j_th] + 1e-8 ) )
                            temp_loss1 += -torch.log(
                                torch.sigmoid(pair_dict_pos[i_th_user][i_th] - pair_dict_neg[i_th_user][j_th]) + margin)
                except:
                    pass

        reg_user = torch.norm(
            self.meta_embeddings_user.matmul(self.meta_embeddings_user.t()) - torch.eye(self.meta_emb_dim_u).to(
                self.device))
        reg_post = torch.norm(
            self.meta_embeddings_post.matmul(self.meta_embeddings_post.t()) - torch.eye(self.meta_emb_dim_p).to(
                self.device))

        if self.control_rate > 0.0:
            a = temp_loss0.cpu().data.numpy()
        else:
            a = 0.0

        if self.control_rate < 1.0:
            b = temp_loss1.cpu().data.numpy()
        else:
            b = 0.0

        ratio0 = (a + b) / (a + 1e-8)
        ratio1 = (a + b) / (b + 1e-8)

        total_loss = self.lambda1 * reg_user + \
                     self.lambda2 * reg_post + \
                     self.control_rate * ratio0 * temp_loss0 + \
                     (1 - self.control_rate) * ratio1 * temp_loss1

        total_loss.backward()
        self.optimizer.step()
        return total_loss.cpu().data.numpy()



    def predict(self, Xi, Xv):
        with torch.no_grad():
            temporary = self.tau
            self.tau = self.tau_test
            model = self.eval()
            output, second_order_embedding_output_list = model(Xi, Xv)
            pred = torch.sigmoid(output).cpu()
            # return pred.data.numpy() > 0.5

            if len(np.array(Xv).shape) >= 2:
                self.tau = temporary
                return [1.0 if i_th.data.numpy() >= 0.5 else 0.0 for i_th in pred], [float(i_th.data.numpy()) for i_th                                                                                     in pred]
            else:
                self.tau = temporary
                return 1.0 if pred.data.numpy() >= 0.5 else 0.0, pred.data.numpy()


if __name__ == "__main__":
    print('hi')