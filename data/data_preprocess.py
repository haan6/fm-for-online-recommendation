# -*- coding:utf-8 -*-

"""
@author: Yeji Han

This script is used to preprocess the raw data file
"""

import random
import torch
from sklearn.datasets import load_svmlight_file
import numpy as np

def load_criteo_category_index(file_path):
    f = open(file_path, 'r')
    cate_dict = []

    for i in range(39):
        cate_dict.append({})

    for line in f:
        data = line.strip().split(',')
        cate_dict[int(data[0])][data[1]] = int(data[2])

    return cate_dict

def read_criteo_data(file_path, emb_file):
    result = {'size': 0, 'label': [], 'index': [], 'value': [], 'feature_sizes':[]}
    cate_dict = load_criteo_category_index(emb_file)

    for item in cate_dict:
        result['feature_sizes'].append(len(item))

    f = open(file_path, 'r')
    for line in f:
        data = line.strip().split(',')
        result['label'].append(int(data[0]))
        indices = [int(item) for item in data[1:]]
        values = [1 for i in range(39)]
        result['index'].append(indices)
        result['value'].append(values)

    result['size'] += len(result['value'])

    return result

def balance_criteo_data(file_path, emb_file):
    result = read_criteo_data(file_path, emb_file)

    indices = {'0': [], '1': []}

    for i in range(len(result['label'])):
        if result['label'][i] == 0:
            indices['0'].append(i)
        else:
            indices['1'].append(i)

    random.shuffle(indices['0'])
    indices['0'] = indices['0'][:len(indices['1'])]

    indices['0'].extend(indices['1'])
    random.shuffle(indices['0'])
    balanced_index = indices['0']

    balanced_Xi = []
    balanced_Xv = []
    balanced_Y = []

    for i in balanced_index:
        balanced_Xi.append(result['index'][i])
        balanced_Xv.append(result['value'][i])
        balanced_Y.append(result['label'][i])

    result['index'] = balanced_Xi
    result['value'] = balanced_Xv
    result['label'] = balanced_Y

    result['size'] = len(result['value'])

    return result

def read_svm_file(file_path, permutation=False):
    X, y = load_svmlight_file(file_path, n_features=8)
    X = X.toarray()

    permutation = False
    if permutation:
        idx = np.random.permutation(X.shape[0])
        x_train_s = np.asarray(X[idx])
        rate_train_s = np.asarray(y[idx])
    else:
        x_train_s = np.asarray(X)
        rate_train_s = np.asarray(y)

    down_sampling = 5
    inputs_matrix = torch.tensor(x_train_s[0:x_train_s.size:down_sampling]).double()
    outputs = torch.tensor(rate_train_s[0:x_train_s.size:down_sampling]).double()