import os
import torch
import numpy as np
import matplotlib
import pickle
import pandas as pd
matplotlib.use('Agg')


from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from models.models_meta_emb.meta_fm import FM_MetaEmbedding


"""
Since FM has limitation in dealing with real dataset where the number of user 
and items is extremely huge, >=10^8, we develop the meta embedding FM which targets the mentioned limitation.
This code conducts the real company dataset which can not be revealed. 
However, we reveal the code and experiment process to introduce how our code can be runned given the processed dataset.
For more detail, refer to models_meta_emb/meta_fm.py
"""


###############################################################################################
# get_binary_representaion
###############################################################################################
def _get_binary_repr(train_Xi):
    output_list = []
    for i_Xi in train_Xi:
        temp_Xi = []

        a = list(np.binary_repr(i_Xi[0], width=binary_repr))
        a += list(np.binary_repr(i_Xi[1], width=binary_repr2))
        a += list(np.binary_repr(i_Xi[2], width=binary_repr2))

        #         a = list(np.binary_repr(i_Xi[1],width = binary_repr2))
        #         a += list(np.binary_repr(i_Xi[2],width = binary_repr2))

        output_list.append([int(i_a) for i_a in a])

    return np.asarray(output_list)




###############################################################################################
# load_dataset
###############################################################################################

train_test_rate = 0.8
save_path = './jupyter/result_dir/'
save_filename = 'Dataset2019-03-15_MetaEmbedding_user_auc_score_version2_numrep1_numexpidx402_numhiddenemb50_tasktypeextra_TrainingTest80'
userhash_path = './pickle_user_prob/'
saved_pickle_file = 'Dataset2019-03-15_userprob_tasktypeextra_TrainingTest80_batchlen10429047'


with open(save_path + save_filename + '.pickle', 'rb') as f:
    data_dict = pickle.load(f)
    train_Xi = data_dict['train_Xi']
    train_Xv = data_dict['train_Xv']
    train_Y = data_dict['train_Y']

    test_Xi = data_dict['test_Xi']
    test_Xv = data_dict['test_Xv']
    test_Y = data_dict['test_Y']
    len_post_type = data_dict['len_post_type']
    num_hidden_embedding = data_dict['num_hidden_embedding']

    print('#' * 30)
    print('data loaded')
    print('train_Xi.shape,train_Y.shape')
    print(train_Xi.shape, train_Y.shape)

    print('test_Xi.shape,test_Y.shape')
    print(test_Xi.shape, test_Y.shape)

    ####################################
    train_shuffled_idx = np.random.choice(int(len(train_Y)), replace=False, size=int(len(train_Y) * 1.0))
    test_shuffled_idx = np.random.choice(int(len(test_Y)), replace=False, size=int(len(test_Y) *  1.0))

    train_Xi = train_Xi[train_shuffled_idx]
    train_Xv = train_Xv[train_shuffled_idx]
    train_Y = train_Y[train_shuffled_idx]

    test_Xi = test_Xi[test_shuffled_idx]
    test_Xv = test_Xv[test_shuffled_idx]
    test_Y = test_Y[test_shuffled_idx]

    ####################################

    print('#' * 30)
    print('shuffled data loaded')
    print('train_Xi.shape,train_Y.shape')
    print(train_Xi.shape, train_Y.shape)
    print('test_Xi.shape,test_Y.shape')
    print(test_Xi.shape, test_Y.shape)

    full_length = len(train_Xi) - 1
    num_trainset = np.shape(train_Xi)[0]



print('loaded_saved_pickle_file')
print(saved_pickle_file)

with open(userhash_path + saved_pickle_file + '.pickle', 'rb') as f:
    result_dict = pickle.load(f)
user_dict_prob = result_dict['user_dict_prob']


###############################################################################################
# experiment_option
###############################################################################################

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

num_rep = 1
num_exp_idx = 403  # experiment index
task_type = 'extra'
num_hidden_embedding = 50
binary_repr = 8 + 1
binary_repr2 = 40 + 1
fm_meta_embedding = FM_MetaEmbedding(len_post_type,
                                     num_hidden_embedding,
                                     [binary_repr, binary_repr2, binary_repr2])

fm_meta_embedding.control_rate =  .5 # for loss regulaizer


###############################################################################################
# run_experiment
###############################################################################################
num_batch = 50000  # 201
num_iteration = 2000
loss_list = []
user_mean_auc_list = []
for i in range(num_iteration + 1):

    sampled_idx = (np.arange(int(num_batch * (i)), int(num_batch * (i + 1))) % full_length)
    binary_train_Xi = _get_binary_repr(train_Xi[sampled_idx])

    pred_prob_list = []
    pred_prob_list2 = []
    for i_th_rows in train_Xi[sampled_idx]:

        #####################################################################################################
        # filters the training dataset by calculated ctr from dataset
        #####################################################################################################
        try:
            if user_dict_prob[int(i_th_rows[1])]['pid_idx'][int(i_th_rows[2])] > 0:
                prob0 = user_dict_prob[int(i_th_rows[1])]['pid_idx'][int(i_th_rows[2])]
            else:
                prob0 = 1e-16
        except:
            prob0 = 1e-16

        try:
            if user_dict_prob[int(i_th_rows[1])]['post_type'][int(i_th_rows[0])] > 0:
                prob2 = user_dict_prob[int(i_th_rows[1])]['post_type'][int(i_th_rows[0])]
            else:
                prob2 = 1e-16
        except:
            prob2 = 1e-16


        if prob0 + prob2 > 2e-16:
            pred_prob_list.append(prob0)
            pred_prob_list2.append(prob2)
        else:
            pred_prob_list.append(-1)
            pred_prob_list2.append(-1)


    pred_prob_list = np.asarray(pred_prob_list)
    pred_prob_list2 = np.asarray(pred_prob_list2)
    partial_idx = np.asarray(pred_prob_list) > 0


    if len(partial_idx) > 0:
        #####################################################################################################
        # train the filtered batch dataset
        #####################################################################################################
        loss = fm_meta_embedding.fit(binary_train_Xi[partial_idx],
                                    train_Xi[sampled_idx[partial_idx]],
                                    train_Y[sampled_idx[partial_idx]],
                                    pred_prob_list[partial_idx],
                                    pred_prob_list2[partial_idx])
        loss_list.append(loss)
    else:
        pass

    if i%250 == 0 and i>0 :

        #####################################################################################################
        # measure user mean auc that averages the AUC for each users
        #####################################################################################################

        pred_label, pred_prob = fm_meta_embedding.predict(binary_train_Xi, train_Xi[sampled_idx])
        user_pred_dict = OrderedDict()
        user_label_dict = OrderedDict()
        zero_mean_auc = []
        nonzero_mean_auc = []
        for i_th, (i_th_U, i_th_label, i_th_pred) in enumerate(zip(train_Xi[sampled_idx], train_Y[sampled_idx], pred_prob)):
            if i_th_U[1] not in user_pred_dict.keys():
                user_pred_dict[i_th_U[1]] = [i_th_pred]
                user_label_dict[i_th_U[1]] = [i_th_label]
            else:
                user_pred_dict[i_th_U[1]].append(i_th_pred)
                user_label_dict[i_th_U[1]].append(i_th_label)


        avg_roc_score_dict = OrderedDict()
        for i_th, i_th_user_keys in enumerate(user_pred_dict):
            if len(user_label_dict[i_th_user_keys]) >= 3:
                try:
                    auc_score = roc_auc_score(np.array(user_label_dict[i_th_user_keys]),
                                              np.array(user_pred_dict[i_th_user_keys]).squeeze())
                    avg_roc_score_dict[i_th_user_keys] = auc_score

                    if auc_score == 0:
                        zero_mean_auc.append(len(user_label_dict[i_th_user_keys]))
                    else:
                        nonzero_mean_auc.append(len(user_label_dict[i_th_user_keys]))
                except:
                    # print(i_th)
                    pass


        # print(avg_roc_score_array.mean())
        avg_roc_score_array = np.asarray(list(avg_roc_score_dict.values()))
        a, b = avg_roc_score_array[avg_roc_score_array > 0].mean(), len(avg_roc_score_array[avg_roc_score_array > 0]),
        c, d = avg_roc_score_array[avg_roc_score_array == 0].mean(), len(avg_roc_score_array[avg_roc_score_array == 0])

        accuracy = len(np.where(pred_label == train_Y[sampled_idx])[0]) / len(train_Y[sampled_idx])
        print('%d iter, loss : %.4f , accuracy : %.4f ,  control_Rate : %.4f' % (i, loss, accuracy, fm_meta_embedding.control_rate))
        print('         user_mean_auc : %.4f , nonzer_auc : %.4f, num_nonz : %d , num_z : %d' % (
        avg_roc_score_array.mean(), a, b, d))

        user_mean_auc_list.append(avg_roc_score_array)
        saved_param_name = './jupyter/torch_save_param/fm_embedding' \
                           + '_numhiddenemb' + str(num_hidden_embedding) \
                           + '_tasktype' + task_type \
                           + '_v' + str(num_exp_idx)\
                           + '_' + str(i) \
                           + '_TrainingTest' + str(int(train_test_rate * 100)) \
                           + '_Rate' + str(int(fm_meta_embedding.control_rate * 100)) + '.pt'

        print('save param_name :' + saved_param_name)
        torch.save(fm_meta_embedding.state_dict(), saved_param_name)
        print('')


        if fm_meta_embedding.tau - 1 >= 1:
            fm_meta_embedding.tau -= 1

        result_record_dict = {'train_uma_list': user_mean_auc_list, 'loss_list': loss_list}

        save_path = './jupyter/result_dir_pickle/'
        save_filename2 = 'fm_embedding' \
                         + '_numhiddenemb' + str(num_hidden_embedding) \
                         + '_tasktype' + task_type \
                         + '_v' + str(num_exp_idx) \
                         + '_' + str(i) \
                         + '_TrainingTest' + str(int(train_test_rate * 100)) \
                         + '_Rate' + str(int(fm_meta_embedding.control_rate * 100))

        with open(save_path + save_filename2 + '.pickle', 'wb') as f:
            pickle.dump(result_record_dict, f)
        print('save_path : %s' % (save_path))
        print('save_record_file : %s' % (save_filename2))

    else:
        user_mean_auc_list.append(-1)



