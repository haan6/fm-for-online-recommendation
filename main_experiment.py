from utils import data_preprocess
import os
import pickle
import numpy as np

import sys
from time import time

sys.path.append('../')

from models.models_online_deep.deepfm_adam import DeepFMAdam
from models.models_online_deep.deepfm_onn import DeepFMOnn
from models.models_online_deep.nfm_adam import NFMAdam
from models.models_online_deep.nfm_onn import NFMOnn
from models.models_online_deep.fm_adam import FMAdam

########################################################################################################################
# save path
########################################################################################################################
save_log = os.getcwd() + '/performance/save_log/'
save_model = os.getcwd() + '/performance/save_model/'

########################################################################################################################
# dataset setup
########################################################################################################################
train_dict = data_preprocess.read_criteo_data('dataset/criteo/tiny_train_input.csv', 'dataset/criteo/category_emb.csv')
train_dict_size = train_dict['size']

num_batchdata = 2500
num_batch = 10
data_config = "Iteration"
# data_config = 3

if data_config == "Iteration":
    batch_train_Xi_list, batch_train_Xv_list, batch_train_Y_list, ratio_list \
        = data_preprocess.create_ten_iter('dataset/criteo/tiny_train_input.csv', 'dataset/criteo/category_emb.csv', num_batch, num_batchdata)

elif isinstance(data_config, int):
    batch_train_Xi_list, batch_train_Xv_list, batch_train_Y_list, ratio_list \
        = data_preprocess.create_dataset('dataset/criteo/tiny_train_input.csv', 'dataset/criteo/category_emb.csv', int(num_batch / data_config), num_batch, num_batchdata)

else:
    batch_train_Xi_list, batch_train_Xv_list, batch_train_Y_list, ratio_list \
        = data_preprocess.create_dataset('dataset/criteo/tiny_train_input.csv', 'dataset/criteo/category_emb.csv', int(num_batch / 2), num_batch, num_batchdata)

########################################################################################################################
# model setup
########################################################################################################################

num_hidden_layers = 5
neuron_per_hidden_layer = 10
data_feature_dim = 39
embedding_size = 10
n = 0.0001

feature_sizes = [63, 113, 126, 51, 224, 148, 100, 79, 104, 9, 32, 57, 82, 1457, 555, 176373, 129683, 305, 19, 11887,
                 632, 3, 41738, 5170, 175446, 3170, 27, 11356, 165602, 10, 4641, 2030, 4, 172761, 18, 15, 57903, 86,
                 44549]
# num_feature = sum(feature_sizes)

model_list = [
    DeepFMAdam(feature_sizes,
               embedding_size=embedding_size,
               num_hidden_layers=num_hidden_layers,
               neuron_per_hidden_layer=neuron_per_hidden_layer,
               n=n),
    DeepFMOnn(feature_sizes,
              embedding_size=embedding_size,
              num_hidden_layers=num_hidden_layers,
              neuron_per_hidden_layer=neuron_per_hidden_layer,
              n=n),
    NFMAdam(feature_sizes,
            embedding_size=embedding_size,
            num_hidden_layers=num_hidden_layers,
            neuron_per_hidden_layer=neuron_per_hidden_layer,
            n=n),
    NFMOnn(feature_sizes,
           embedding_size=embedding_size,
           num_hidden_layers=num_hidden_layers,
           neuron_per_hidden_layer=neuron_per_hidden_layer,
           n=n),
    FMAdam(feature_sizes,
           embedding_size=embedding_size,
           n=n)
]
model_name_list = [str(model).split('-')[0] for model in model_list]
print(model_name_list)

########################################################################################################################
# pre-training
########################################################################################################################
for ith_model, ith_model_name in zip(model_list, model_name_list):
    print(f"====={ith_model_name}=====")
    for j in range(1000):
        loss_emb = ith_model.update_embedding(batch_train_Xi_list[int(num_batch/2)],
                                              batch_train_Xv_list[int(num_batch/2)],
                                              batch_train_Y_list[int(num_batch/2)])
        pred_label = ith_model.predict(batch_train_Xi_list[int(num_batch/2)],
                                       batch_train_Xv_list[int(num_batch/2)])

        if j % 100 == 0:
            print('i th iter %d , loss : %f' % (j, loss_emb.cpu().data))
            right_count = len((np.where(np.asarray(pred_label) == np.asarray(batch_train_Y_list[int(num_batch/2)])))[0])
            total_count = len(np.asarray(batch_train_Y_list[int(num_batch/2)]))
            print('training accuracy : %.4f\n' % (right_count / total_count))

########################################################################################################################
# run_main_exp
########################################################################################################################

result_dict = {}
result_dict['roc'] = {}
result_dict['data_ratio'] = {}
result_dict['time'] = {}
result_dict['accuracy'] = {}

result_dict['num_batch'] = num_batch
result_dict['num_batchdata'] = num_batchdata
result_dict['user_auc_mean'] = {}

for ith_exp in range(num_batch):
    print('#' * 100)

    for jth_model_name, jth_model in zip(model_name_list, model_list):
        print('%d th batch, %s model' % (ith_exp + 1, jth_model_name))
        print('neg ratio : %d,  pos ratio %d ' % (ratio_list[ith_exp][0], ratio_list[ith_exp][1]))

        time_elapsed, accuracy, roc, confusion_matrix\
            = jth_model.run_experiment(batch_train_Xi_list[ith_exp], batch_train_Xv_list[ith_exp], batch_train_Y_list[ith_exp])

        print('fpr : %.4f , tpr : %.4f ' % (roc['fpr'], roc['tpr']))
        print('confusion matrix : %s' % confusion_matrix)
        print('accuracy : %.4f \n' % accuracy)

        if ith_exp == 0:
            result_dict['roc'][jth_model_name] = [roc]
            result_dict['data_ratio'][jth_model_name] = [ratio_list[ith_exp]]
            result_dict['time'][jth_model_name] = [time_elapsed]
            result_dict['accuracy'][jth_model_name] = [accuracy]

        else:
            result_dict['roc'][jth_model_name].append(roc)
            result_dict['data_ratio'][jth_model_name].append(ratio_list[ith_exp])
            result_dict['time'][jth_model_name].append(time_elapsed)
            result_dict['accuracy'][jth_model_name].append(accuracy)

timestamp = time()

save_filename = 'Time_Stamp' + str(timestamp)\
                + '-Dataset' + str('criteo') \
                + '-Num_BatchLength' + str(num_batchdata) \
                + '-Num_Batch' + str(num_batch) \
                + '-Num_Hidden_Layers' + str(num_hidden_layers) \
                + '-Neuron_Per_Hidden_Layer' + str(neuron_per_hidden_layer) \
                + '_' + str(data_config)

with open(save_log + save_filename + '.pickle', 'wb') as f:
    pickle.dump(result_dict, f)

for ith_model, ith_model_name in zip(model_list, model_name_list):
    with open(save_model + ith_model_name + '_' + str(data_config) + '.pickle', 'wb') as f:
        pickle.dump(ith_model, f)

print('save_log : %s' % (save_log))
print('save_model : %s' % (save_model))
