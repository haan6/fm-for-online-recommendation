import os
import pickle
import torch

from model.FM import FM
from model.SGD_NFM import SGD_NFM
from model.ONN_NFM import ONN_NFM,_make_onnnfm_model

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

num_batchdata = 1000
num_batch = 20
batch_train_Xi_list, batch_train_Xv_list, batch_train_Y_list, ratio_list = \
        data_preprocess._construct_batch_criteo_data(train_dict, num_batchdata, num_batch)

########################################################################################################################
# model setup
########################################################################################################################

max_num_hidden_layer = 10
qtd_neuron_per_hidden_layer = 10
data_feature_dim = 39
embedding_siz = 10
lr_onn = 0.01
onn_nfm_instance, option_name = _make_onnnfm_model(data_feature_dim,
                                                   train_dict['feature_sizes'],
                                                   max_num_hidden_layer,
                                                   qtd_neuron_per_hidden_layer,
                                                   embedding_siz,
                                                   lr_onn,
                                                   'OnnNfm')
print(option_name)
########################################################################################################################
# model_option_setup
########################################################################################################################

# #####################################################
# #option1
# #####################################################
# max_num_hidden_layer_list = [10,10,10,10]
# qtd_neuron_per_hidden_layer_list = [10,10,10,10]
# embedding_siz_list = [5,10,30,50]
# lr_onn_list = [.01,.01,.01,.01]


# #####################################################
# #option2
# #####################################################
# max_num_hidden_layer_list = [2,5,10,20]
# qtd_neuron_per_hidden_layer_list = [10,10,10,10]
# embedding_siz_list = [5,5,5,5]
# lr_onn_list = [.01,.01,.01,.01]

# #####################################################
# #option3
# #####################################################
# max_num_hidden_layer_list = [10,10,10,10]
# qtd_neuron_per_hidden_layer_list = [10,10,10,10]
# embedding_siz_list = [5,5,5,5]
# lr_onn_list = [.1,.01,.001,.0001]


#####################################################
#option4
#####################################################
max_num_hidden_layer_list = [10,10,10,10]
qtd_neuron_per_hidden_layer_list = [5,10,20,50]
embedding_siz_list = [5,5,5,5]
lr_onn_list = [.01,.01,.01,.01]


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
result_dict['num_batch'] = num_batch
result_dict['num_batchdata'] = num_batchdata





print(batch_train_Xi_list[0].shape)







onn_nfm_instance_list = []
onn_nfm_option_list = []
for i_th_exp in range(num_batch):

    print('')
    #for j_th_model_name,j_th_model in zip(model_name_list,model_list):
    # batch_train_Xi_list,batch_train_Xv_list,batch_train_Y_list,ratio_list
    for j_th,(j_th_numhidden,j_th_qtd,j_th_embedding,j_th_lr) in enumerate(zip(max_num_hidden_layer_list,qtd_neuron_per_hidden_layer_list,embedding_siz_list,lr_onn_list)):
        if i_th_exp == 0:
            onn_nfm_instance, option_name = _make_onnnfm_model(data_feature_dim,
                                                               train_dict['feature_sizes'],
                                                               j_th_numhidden,
                                                               j_th_qtd,
                                                               j_th_embedding,
                                                               j_th_lr,
                                                               'OnnNfm')
            onn_nfm_instance_list.append(onn_nfm_instance)
            onn_nfm_option_list.append(option_name)


            print('%d th batch , %s model' % (i_th_exp + 1,onn_nfm_option_list[j_th]))
            print('neg ratio : %d  ,  pos ratio %d ' % (ratio_list[i_th_exp][0], ratio_list[i_th_exp][1]))

            # roc, confusion_matrix, time_elapsed = onn_nfm.run_batch_exp(batch_train_Xi_list[i_th_exp],
            #                                                             batch_train_Xv_list[i_th_exp],
            #                                                             batch_train_Y_list[i_th_exp])
            roc, confusion_matrix, time_elapsed = onn_nfm_instance.run_batch_exp(batch_train_Xi_list[i_th_exp],
                                                                                 batch_train_Xv_list[i_th_exp],
                                                                                 batch_train_Y_list[i_th_exp])

            print('fpr : %.4f , tpr : %.4f ' % (roc['fpr'], roc['tpr']))


            ################################################################################################
            # save result_dict

            result_dict['roc'][option_name] = [roc]
            result_dict['data_ratio'][option_name] = [ratio_list[i_th_exp]]
            result_dict['time'][option_name] = [time_elapsed]
        else :
            print('%d th batch , %s model' % (i_th_exp + 1, onn_nfm_option_list[j_th]))
            print('neg ratio : %d  ,  pos ratio %d ' % (ratio_list[i_th_exp][0], ratio_list[i_th_exp][1]))

            # roc, confusion_matrix, time_elapsed = onn_nfm.run_batch_exp(batch_train_Xi_list[i_th_exp],
            #                                                             batch_train_Xv_list[i_th_exp],
            #                                                             batch_train_Y_list[i_th_exp])

            roc, confusion_matrix, time_elapsed = onn_nfm_instance_list[j_th].run_batch_exp(batch_train_Xi_list[i_th_exp],
                                                                                             batch_train_Xv_list[i_th_exp],
                                                                                             batch_train_Y_list[i_th_exp])

            print('fpr : %.4f , tpr : %.4f ' % (roc['fpr'], roc['tpr']))


            ################################################################################################
            # save result_dict
            result_dict['roc'][onn_nfm_option_list[j_th]].append(roc)
            result_dict['data_ratio'][onn_nfm_option_list[j_th]].append(ratio_list[i_th_exp])
            result_dict['time'][onn_nfm_option_list[j_th]].append(time_elapsed)




# save_filename = 'Dataset' + str('criteo') \
#                 + '_Numbatchlen' + str(num_batchdata) \
#                 + '_Numbatch' + str(num_batch) \
#                 + '_ModelList' + 'OnnNfm' \
#                 + '_HiddenLayer' + str(max_num_hidden_layer) \
#                 + '_QtdSecondLayer' + str(qtd_neuron_per_hidden_layer) \
#                 + '_lr' + str(lr_onn)

save_filename = 'Dataset' + str('criteo') \
                + '_Numbatchlen' + str(num_batchdata) \
                + '_Numbatch' + str(num_batch) \
                + '_ModelList' + 'OnnNfm' \
                + '_Purpose' + 'LearningRate'
                #+ '_Purpose' + 'HiddenLayerDepth'
                #+ '_Purpose' + 'EmbeddingSizComparison'



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