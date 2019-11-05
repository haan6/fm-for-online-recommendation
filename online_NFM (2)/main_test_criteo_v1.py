import os
import pickle
import torch

from model.FM import FM
from model.SGD_NFM import SGD_NFM
from model.ONN_NFM import ONN_NFM,_make_onnnfm_model,roc_auc_score

import sys
import datetime

sys.path.append('../')
from utils import data_preprocess
import matplotlib.pyplot as plt

from builtins import str


########################################################################################################################
# save path
########################################################################################################################
save_path = os.getcwd() + '/result_dir/'
os.mkdir(save_path) if not os.path.exists(save_path) else 1



########################################################################################################################
# data setup
########################################################################################################################


train_dict = data_preprocess.read_criteo_data('data/criteo/tiny_train_input.csv', 'data/criteo/category_emb.csv')
train_dict_size = train_dict['size']
# train_Xi, train_Xv, train_Y \
#     = train_dict['index'][:int(train_dict_size * 0.05)], \
#       train_dict['value'][:int(train_dict_size * 0.05)], \
#       train_dict['label'][:int(train_dict_size * 0.05)]

num_batchdata = 1000
num_batch = 20
batch_train_Xi_list, batch_train_Xv_list, batch_train_Y_list, ratio_list = data_preprocess._construct_batch_criteo_data(
    train_dict, num_batchdata, num_batch)

########################################################################################################################
# model setup
########################################################################################################################
# sgd_nfm = SGD_NFM(39,
#                   train_dict['feature_sizes'],
#                   5,
#                   10,
#                   batch_size=20)
# models = [(fm, "FM"), (sgd_nfm, "SGD_NFM"), (onn_nfm, "ONN_NFM")]


# max_num_hidden_layer = 10
# qtd_neuron_per_hidden_layer = 10
# data_feature_dim = 39
# embedding_siz = 10
# lr_onn = 0.01
# onn_nfm_instance, option_name = _make_onnnfm_model(data_feature_dim,
#                                                    train_dict['feature_sizes'],
#                                                    max_num_hidden_layer,
#                                                    qtd_neuron_per_hidden_layer,
#                                                    embedding_siz,
#                                                    lr_onn,
#                                                    'onn_nfm')

# def _make_nfm_option(num_hidden_layer,
#                      qtd_neuron_per_hidden_layer,
#                      embedding_siz,
#                      lr,
#                      model_name):
#     # assert(isinstance(model_name, str))
#     # option_name = model_name
#     # option_name += '_NumHiddenLayer' +str(num_hidden_layer)
#     # option_name += '_QtdNumNueron' +str(qtd_neuron_per_hidden_layer)
#     # option_name += '_EmbeddingSiz' +str(embedding_siz)
#     # option_name += '_lr' +str(lr)
#
#     option_dict = {}
#     option_dict['num_hidden_layer'] = num_hidden_layer
#     option_dict['qtd_neuron_per_hidden_layer'] = qtd_neuron_per_hidden_layer
#     option_dict['embedding_siz'] = embedding_siz
#     option_dict['lr'] = lr
#
#     return option_dict,option_name










max_num_hidden_layer = 10
qtd_neuron_per_hidden_layer = 10
data_feature_dim = 39
embedding_siz = 10
lr_onn2 = 0.01
onn_nfm_lr0001 = ONN_NFM(data_feature_dim,
                          train_dict['feature_sizes'],
                          max_num_hidden_layers=max_num_hidden_layer,
                          qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                          dropout_shallow=[0.5],
                          embedding_size=embedding_siz,
                          n_classes=2,
                          batch_size=1,
                          verbose=False,
                          interaction_type=True,
                          eval_metric=roc_auc_score,
                          b=0.99,
                          n=0.0001,
                          s=0.2,
                          use_cuda=True,
                          greater_is_better=True)


sgd_nfm001 =  SGD_NFM(data_feature_dim,
                      train_dict['feature_sizes'],
                      max_num_hidden_layers=max_num_hidden_layer,
                      qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                      dropout_shallow=[0.5],
                      embedding_size=embedding_siz,
                      n_classes=2,
                      batch_size=1,
                      loss_type = 'logloss',
                      verbose=False,
                      interaction_type=True,
                      eval_metric=roc_auc_score,
                      b=0.99,
                      n=0.001,
                      s=0.2,
                      use_cuda=True,
                      greater_is_better=True)


fm_b50 = FM(data_feature_dim,
        train_dict['feature_sizes'],
        batch_size=50)

model_list = [sgd_nfm001,onn_nfm_lr0001,fm_b50]
model_name_list = ['sgd_nfm001','onn_nfm0001','fm_b50']





########################################################################################################################
# run_main_exp
########################################################################################################################

# num_batchdata = 500
# num_batch= 10
# batch_train_Xi_list,batch_train_Xv_list,batch_train_Y_list,ratio_list  = data_preprocess._construct_batch_criteo_data(train_dict,num_batchdata,num_batch)


# batch_train_Xi_list,batch_train_Xv_list,batch_train_Y_list,ratio_list

result_dict = {}
result_dict['roc'] = {}
result_dict['data_ratio'] = {}
result_dict['time'] = {}
for i_th_exp in range(num_batch):

    print('')
    for j_th_model_name,j_th_model in zip(model_name_list,model_list):
    # batch_train_Xi_list,batch_train_Xv_list,batch_train_Y_list,ratio_list



        print('%d th batch , %s model' % (i_th_exp + 1,j_th_model_name))
        print('neg ratio : %d  ,  pos ratio %d ' % (ratio_list[i_th_exp][0], ratio_list[i_th_exp][1]))

        # roc, confusion_matrix, time_elapsed = onn_nfm.run_batch_exp(batch_train_Xi_list[i_th_exp],
        #                                                             batch_train_Xv_list[i_th_exp],
        #                                                             batch_train_Y_list[i_th_exp])

        roc, confusion_matrix, time_elapsed = j_th_model.run_batch_exp(batch_train_Xi_list[i_th_exp],
                                                                    batch_train_Xv_list[i_th_exp],
                                                                    batch_train_Y_list[i_th_exp])

        print('fpr : %.4f , tpr : %.4f ' % (roc['fpr'], roc['tpr']))


        if i_th_exp == 0:
            # result_dict['roc'] = [roc]
            # result_dict['data_ratio'] = [ratio_list[i_th_exp]]
            # result_dict['time'] = [time_elapsed]
            result_dict['roc'][j_th_model_name ] = [roc]
            result_dict['data_ratio'][j_th_model_name ] = [ratio_list[i_th_exp]]
            result_dict['time'][j_th_model_name] = [time_elapsed]

        else:
            # result_dict['roc'].append(roc)
            # result_dict['data_ratio'].append(ratio_list[i_th_exp])
            # result_dict['time'].append(time_elapsed)

            result_dict['roc'][j_th_model_name].append(roc)
            result_dict['data_ratio'][j_th_model_name].append(ratio_list[i_th_exp])
            result_dict['time'][j_th_model_name].append(time_elapsed)

# save_filename = 'Dataset' + str('criteo') \
#                 + '_Numbatchlen' + str(num_batchdata) \
#                 + '_Numbatch' + str(num_batch) \
#                 + '_ModelList' + str(model_name_list)\
#                 + '_HiddenLayer' + str(max_num_hidden_layer) \
#                 + '_QtdSecondLayer' + str(qtd_neuron_per_hidden_layer) \
#                 + '_lr' + str(lr_onn)


save_filename = 'Dataset' + str('criteo') \
                + '_Numbatchlen' + str(num_batchdata) \
                + '_Numbatch' + str(num_batch) \
                + '_ModelList' + 'Full' \
                + '_HiddenLayer' + str(max_num_hidden_layer) \
                + '_QtdSecondLayer' + str(qtd_neuron_per_hidden_layer) \


with open(save_path + save_filename + '.pickle', 'wb') as f:
    pickle.dump(result_dict, f)


print('')
print('save_path : %s'%(save_path))
print('save_file : %s'%(save_filename))

# with torch.cuda.device(0):
#     time_elapsed = {"FM": 0, "SGD_NFM": 0, "ONN_NFM": 0}
#     roc_scores = {"FM": [], "SGD_NFM": [], "ONN_NFM": []}
#
#     print("===== Instantiating Models =====")
#
#
#     print("===== Models are Ready =====")

# for i, model in enumerate(models):
#     print(f"===== Training {models[i][1]} =====")
#     time_elapsed[models[i][1]], roc_scores[models[i][1]] = models[0].evaluate(train_Xi, train_Xv, train_Y)
#     print(time_elapsed[models[i][1]], roc_scores[models[i][1]])
#     print(f"===== Evaluating {models[i][1]} is Finished. Time: {time_elapsed[models[i][1]]} =====")
#
# time_elapsed[models[2][1]], roc_scores[models[2][1]] = models[2][0].evaluate(train_Xi, train_Xv, train_Y)


# print("===== Drawing a plot =====")
#
# now = datetime.datetime.now()
# date = now.strftime('%Y-%m-%d')
#
# fig, ax = plt.subplots()
#
# ax.ylim(-0.04, 1.04)
# ax.xlabel('FPR')
# ax.ylabel('TPR')
#
# for i, (mark, color) in enumerate(zip(
#         ['s', 'o', 'v'], ['r', 'g', 'b'])):
#     tpr = []
#     fpr = []
#
#     for j in range(roc_scores[models[i][1]]):
#         tpr.append([roc_scores[models[i][1]][j]["tpr"]])
#         fpr.append([roc_scores[models[i][1]][j]["fpr"]])
#         ax.plot(fpr, tpr, color=color,
#                 marker=mark,
#                 markerfacecolor='None',
#                 markeredgecolor=color,
#                 linestyle='None',
#                 label=models[i][0])
#
# ax.set_aspect('equal')
# ax.title('ROC Score')
# ax.legend(loc='lower right')
# fig.savefig(f'{date}_roc_score.png')
#
# plt.show()