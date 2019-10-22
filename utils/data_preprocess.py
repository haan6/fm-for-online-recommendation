# -*- coding:utf-8 -*-

"""
@author: Yeji Han

This script is used to preprocess the raw data file
"""

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