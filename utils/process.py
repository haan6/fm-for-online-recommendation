import os

import pandas as pd
import numpy as np

def get_criteo_dataset():
    continuous_features = 13

    result = {'size': 0, 'label': [], 'index': [], 'value': [], 'feature_sizes': []}
    filepath = f"/dataset/criteo/"
    data = pd.read_csv(os.path.join(filepath, 'train.txt'))

    train_data = data.iloc[:, :-1].values
    Xi_continuous = np.zeros_like(train_data)
    Xi_categorical = data[:, continuous_features:]

    result['label'] = data.iloc[:, -1].values.tolist()



    feature_sizes = np.loadtxt('./dataset/feature_sizes.txt', delimiter=',')
    result['feature_sizes'] = [int(x) for x in feature_sizes]
    result = len(train_data)
