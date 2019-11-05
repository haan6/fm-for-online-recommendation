# -*- coding:utf-8 -*-

"""
@author: Yeji Han

This script is used to preprocess the raw data file
"""
import sys
import numpy as np
import datetime

sys.path.append('../')



def load_criteo_category_index(file_path):
    f = open(file_path, 'r')
    cate_dict = []

    for i in range(39):
        cate_dict.append({})

    for line in f:
        #print(line)
        data = line.strip().split(',')
        cate_dict[int(data[0])][data[1]] = int(data[2])
    return cate_dict



#result = {'size': 0, 'label': [], 'index': [], 'value': [], 'feature_sizes':[]}
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


# idx = 1
# num_batchdata = 5000
# num_batch = 10
def _construct_batch_criteo_data(train_dict,num_batchdata,num_batch):
    batch_train_Xi_list = []
    batch_train_Xv_list = []
    batch_train_Y_list = []
    ratio_list = []


    for idx in range(1, num_batch + 1):
        train_Xi = np.asarray(train_dict['index'][(idx - 1) * num_batchdata:idx * num_batchdata])
        train_Xv = np.asarray(train_dict['value'][(idx - 1) * num_batchdata:idx * num_batchdata])
        train_Y = np.asarray(train_dict['label'][(idx - 1) * num_batchdata:idx * num_batchdata])

        array1 = np.where(np.asarray(train_Y) == 1)[0]
        num_select1 = len(array1)

        sample_ratio = np.random.rand()
        if sample_ratio >= 0.1 and sample_ratio <= 0.9:
            num_select0 = int((num_batchdata - num_select1) * sample_ratio)
        else:
            if sample_ratio < 0.1:
                num_select0 = int((num_batchdata - num_select1) * .1)
            else:
                num_select0 = int((num_batchdata - num_select1) * .9)

        print(num_select0)
        array0 = np.sort(np.asarray(np.random.choice(np.where(np.asarray(train_Y) == 0)[0], num_select0, replace=False)))

        chosen_idx = np.sort(np.concatenate([array0, array1], axis=0))

        batch_train_Xi_list.append(train_Xi[chosen_idx, :])
        batch_train_Xv_list.append(train_Xv[chosen_idx, :])
        batch_train_Y_list.append(train_Y[chosen_idx])

        ratio_list.append([num_select0, num_select1])

    return batch_train_Xi_list,batch_train_Xv_list,batch_train_Y_list,ratio_list

    #     batch_train_Xi_list.append(train_dict['index'][(idx-1)*num_batchdata:idx*num_batchdata])
    #     batch_train_Xv_list.append(train_dict['value'][(idx-1)*num_batchdata:idx*num_batchdata])
    #     batch_train_Y_list.append(train_dict['label'][(idx-1)*num_batchdata:idx*num_batchdata])




