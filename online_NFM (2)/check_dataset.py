import torch
from model.FM import *
from model.SGD_NFM import *
from model.ONN_NFM import *

import sys
import datetime

sys.path.append('../')
from utils import data_preprocess

import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("===== Importing Dataset =====")

    train_dict = data_preprocess.read_criteo_data('data/criteo/tiny_train_input.csv', 'data/criteo/category_emb.csv')
    train_dict_size = train_dict['size']

    print(train_dict.keys())
    print(train_dict['size'])
    print(train_dict['index'][0])
    print(train_dict['feature_sizes'])
    # print(len(train_dict['index'][0]))
    # print(len(train_dict['feature_sizes']))
    print(sum(train_dict['feature_sizes']))



# train_Xi, train_Xv, train_Y = train_dict['index'][:int(train_dict_size * 0.05)], \
#       train_dict['value'][:int(train_dict_size * 0.05)], \
#       train_dict['label'][:int(train_dict_size * 0.05)]


# num_batchdata = 5000
# num_batch= 10
# batch_train_Xi_list,batch_train_Xv_list,batch_train_Y_list,ratio_list  = data_preprocess._construct_batch_criteo_data(train_dict,num_batchdata,num_batch)
