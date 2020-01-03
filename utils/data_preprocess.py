# -*- coding:utf-8 -*-

"""
@author: Yeji Han

This script is used to preprocess the raw data file
"""

import random
import copy
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
    result = {'size': 0, 'label': [], 'index': [], 'value': [], 'feature_sizes': []}
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
    result = {'size': 0, 'label': [], 'index': [], 'value': [], 'feature_sizes': []}

    X, y = load_svmlight_file(file_path, n_features=8)
    X = X.toarray()

    if permutation:
        idx = np.random.permutation(X.shape[0])
        x_train_s = np.asarray(X[idx])
        rate_train_s = np.asarray(y[idx])
    else:
        x_train_s = np.asarray(X)
        rate_train_s = np.asarray(y).astype(int)

    feature_sizes = {str(i): [] for i in range(1, 9)}
    x_train_i = copy.deepcopy(x_train_s)
    for xi in x_train_i:
        for i, emb in enumerate(xi):
            try:
                xi[i] = feature_sizes[str(i + 1)].index(emb)
            except ValueError:
                feature_sizes[str(i + 1)].append(emb)
                xi[i] = feature_sizes[str(i + 1)].index(emb)

    result['size'] = len(x_train_s)
    result['label'] = np.where(rate_train_s == -1, 0, rate_train_s).astype(int)
    result['index'] = x_train_i.astype(int)
    result['value'] = x_train_s
    result['feature_sizes'] = np.asarray([len(feature) for feature in feature_sizes.values()]).astype(int)
    return result


def balance_svm_data(file_path):
    result = read_svm_file(file_path)

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

    result['index'] = np.asarray(balanced_Xi)
    result['value'] = np.asarray(balanced_Xv)
    result['label'] = np.asarray(balanced_Y)

    result['size'] = len(result['value'])

    return result


def _construct_batch_criteo_data(train_dict, num_batchdata, num_batch):
    batch_train_Xi_list = []
    batch_train_Xv_list = []
    batch_train_Y_list = []
    ratio_list = []

    for i in range(num_batch):
        Xi_list = []
        Xv_list = []
        Y_list = []

        for j in range(i * num_batchdata, (i+1) * num_batchdata):
            Xi_list.append(train_dict['index'][j])
            Xv_list.append(train_dict['value'][j])
            Y_list.append(train_dict['label'][j])

        pos = sum(Y_list)
        ratio_list.append((len(Y_list) - pos, pos))

        batch_train_Xi_list.append(Xi_list)
        batch_train_Xv_list.append(Xv_list)
        batch_train_Y_list.append(Y_list)

    return batch_train_Xi_list, batch_train_Xv_list, batch_train_Y_list, ratio_list


def _find_pos_and_neg(file_path, emb_file):
    result = read_criteo_data(file_path, emb_file)
    ratios = {'0': [], '1': []}

    for i, label in enumerate(result['label']):
        ratios[str(label)].append(i)

    return result, ratios


def create_ten_iter(file_path, emb_file, num_batch, num_batchdata):
    result, ratios = _find_pos_and_neg(file_path, emb_file)

    batch_train_Xi_list = []
    batch_train_Xv_list = []
    batch_train_Y_list = []
    ratio_list = []

    for i in range(num_batch):
        Xi_list = []
        Xv_list = []
        Y_list = []

        num_pos = int(num_batchdata / num_batch * (i + 1))
        num_neg = num_batchdata - num_pos

        ratio_list.append((num_neg, num_pos))

        pos_indices = ratios['1'][:num_pos]
        neg_indices = ratios['0'][:num_neg]

        ratios['1'] = ratios['1'][num_pos:]
        ratios['0'] = ratios['0'][num_neg:]

        indices = pos_indices + neg_indices
        random.shuffle(indices)

        for i in indices:
            Xi_list.append(result['index'][i])
            Xv_list.append(result['value'][i])
            Y_list.append(result['label'][i])

        batch_train_Xi_list.append(Xi_list)
        batch_train_Xv_list.append(Xv_list)
        batch_train_Y_list.append(Y_list)

    return batch_train_Xi_list, batch_train_Xv_list, batch_train_Y_list, ratio_list


def create_dataset(file_path, emb_file, batch_ratio, num_batch, num_batchdata):
    result, ratios = _find_pos_and_neg(file_path, emb_file)

    batch_train_Xi_list = []
    batch_train_Xv_list = []
    batch_train_Y_list = []
    ratio_list = []

    for i in range(num_batch):
        Xi_list = []
        Xv_list = []
        Y_list = []
        ratio_list.append((batch_ratio, num_batch-batch_ratio))

        num_pos = int(num_batchdata / num_batch * batch_ratio)
        num_neg = num_batchdata - num_pos

        pos_indices = ratios['1'][:num_pos]
        neg_indices = ratios['0'][:num_neg]

        ratios['1'] = ratios['1'][num_pos:]
        ratios['0'] = ratios['0'][num_neg:]

        indices = pos_indices + neg_indices
        random.shuffle(indices)

        for i in indices:
            Xi_list.append(result['index'][i])
            Xv_list.append(result['value'][i])
            Y_list.append(result['label'][i])

        batch_train_Xi_list.append(Xi_list)
        batch_train_Xv_list.append(Xv_list)
        batch_train_Y_list.append(Y_list)

    return batch_train_Xi_list, batch_train_Xv_list, batch_train_Y_list, ratio_list

def _make_user_post_dict(train_Xi, train_Y):
    pass
