import numpy as np
import scipy.io as sio
from scipy.sparse import lil_matrix
from scipy.sparse import hstack

from datetime import timezone, timedelta, datetime
import csv

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler

import matplotlib

matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


def load_dataset_movielens(filename, lines, columns, nbUsers):
    # Features are one-hot encoded in a sparse matrix

    X = lil_matrix((lines, columns)).astype('float32')
    X2 = lil_matrix((lines, columns + 1)).astype('float64')
    # Labels are stored in a vector
    Y = []
    Y2 = []
    date_time = []
    line = 0
    with open(filename, 'r') as f:
        samples = csv.reader(f, delimiter='\t')
        for userId, movieId, rating, timestamp in samples:
            X[line, int(userId) - 1] = 1
            X[line, int(nbUsers) + int(movieId) - 1] = 1

            X2[line, int(userId) - 1] = 1
            X2[line, int(nbUsers) + int(movieId) - 1] = 1
            X2[line, columns] = 1

            if int(rating) >= 3:
                Y.append(1)
            else:
                Y.append(0)
            line = line + 1
            date_time.append(timestamp)
            Y2.append(rating)

    date_time = np.array(date_time).astype('float32')
    Y = np.array(Y).astype('float64')
    return X, X2, Y, Y2, date_time


def sort_dataset_movielens(X, Y, utc_time_stamp):
    # make up some data
    # time_order = np.array([datetime.utcfromtimestamp(current_utc) for current_utc in utc_time_stamp])
    time_order = np.array([datetime.utcfromtimestamp(current_utc) for current_utc in utc_time_stamp])
    Y = np.array([float(i) for i in Y])

    sorted_X = X[time_order.argsort()]
    sorted_Y = Y[time_order.argsort()]
    return sorted_X, sorted_Y, time_order


def load_dataset_YearPredictionMSD(dataname, isTransformY=False, isRemoveEmpty=False):
    # filename = home_dir + 'Resource/' + dataname
    X, y = load_svmlight_file(dataname)
    X = np.asarray(X.todense())
    n, d = X.shape
    print('Size of X is ' + str(n) + '-by-' + str(d))
    print('Size of y is ' + str(y.shape))
    X = MinMaxScaler().fit_transform(X)

    if isTransformY:
        y = (y * 2) - 3

    if isRemoveEmpty:
        sumX = np.sum(np.abs(X), axis=1)
        idx = (sumX > 1e-6)
        X = X[idx, :]
        y = y[idx]

    return X, y


def load_dataset_fappe(file, logloss_opt=False):
    # read a data file. For a row, the first column goes into Y_;
    # the other columns become a row in X_ and entries are maped to indexs in self.features

    # read feature
    features = {}
    i = len(features)

    f = open(file)
    line = f.readline()
    while line:
        items = line.strip().split(' ')
        for item in items[1:]:
            if item not in features:
                features[item] = i
                i = i + 1
        line = f.readline()
    f.close()

    # data_load
    f = open(file)
    X_ = []
    Y_ = []
    Y_for_logloss = []
    line = f.readline()
    while line:
        items = line.strip().split(' ')
        Y_.append(1.0 * float(items[0]))

        if float(items[0]) > 0:  # > 0 as 1; others as 0
            v = 1.0
        else:
            v = 0.0
        Y_for_logloss.append(v)
        X_.append([features[item] for item in items[1:]])
        line = f.readline()
    f.close()

    Data_Dic = {}
    X_lens = [len(line) for line in X_]
    indexs = np.argsort(X_lens)

    # if logloss_opt == True :
    #     Data_Dic['Y'] = [Y_for_logloss[i] for i in indexs]
    #     Data_Dic['X'] = [X_[i] for i in indexs]
    # else :
    #     Data_Dic['Y'] = [Y_[i] for i in indexs]
    #     Data_Dic['X'] = [X_[i] for i in indexs]

    Data_Dic['Y2'] = [Y_for_logloss[i] for i in indexs]
    Data_Dic['Y'] = [Y_[i] for i in indexs]
    Data_Dic['X'] = [X_[i] for i in indexs]

    return np.asarray(Data_Dic['X']), np.asarray(Data_Dic['Y'])
    # return np.asarray(Data_Dic['X']),np.asarray(Data_Dic['Y']),np.asarray(Data_Dic['Y2'])


if __name__ == "__main__":


    nbUsers = 943
    nbMovies = 1682
    nbFeatures = nbUsers + nbMovies
    nbRatingsTrain = 90570
    nbRatingsTest = 9430

    #print(os.getcwd())
    data_dir = './../dataset/ml-100k/'

    print(data_dir)


    #filename1, filename2 = 'ub.base', 'ub.test'
    filename1, filename2 = 'ua.base', 'ua.test'

    # load dataset
    _, x_train, y_train, rate_train, timestamp_train = load_dataset_movielens(data_dir + filename1,
                                                                              nbRatingsTrain,
                                                                              nbFeatures,
                                                                              nbUsers)

    print(x_train)
    # filename = './../dataset/YearPredictionMSD/YearPredictionMSD'
    # X, y = load_dataset_YearPredictionMSD(filename)


    # filename = './../dataset/YearPredictionMSD/YearPredictionMSD'
    # X, y = load_dataset_YearPredictionMSD(filename)

    # filename = './../dataset/frappe/frappe.train.libfm'
    # Dataset = load_dataset_fappe(filename, logloss_opt=False)

    #print(Dataset['Y'])