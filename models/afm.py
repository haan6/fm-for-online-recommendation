# -*- coding:utf-8 -*-

from utils import data_preprocess
from model import AFM
import torch

result_dict = data_preprocess.read_criteo_data('../data/tiny_train_input.csv', '../data/category_emb.csv')
test_dict = data_preprocess.read_criteo_data('../data/tiny_test_input.csv', '../data/category_emb.csv')

size = len(test_dict['index'])

afm = AFM.AFM(39, result_dict['feature_sizes'], embedding_size = 4, attention_size = 4,is_shallow_dropout = True, dropout_shallow = [0.5],
                 is_attention_dropout = True, dropout_attention=[0.5],
                 attention_layers_activation = 'relu', n_epochs = 64, batch_size = 256, learning_rate = 0.003,
                 optimizer_type = 'adam', is_batch_norm = False, verbose = False, random_seed = 990211, weight_decay = 0.0,
                 use_fm = True, use_ffm = False, loss_type = 'logloss', use_cuda = True, n_class = 1, greater_is_better = True).cuda()
AFM.fit(afm, result_dict['index'], result_dict['value'], result_dict['label'],
        test_dict['index'][int(size*0.75):], test_dict['value'][int(size*0.75):], test_dict['label'][int(size*0.75):], early_stopping=True, save_path='./save_pretrained_model_afm')
# afm.load_state_dict(torch.load('./save_pretrained_model_afm'))
AFM.online_fit(afm, test_dict['index'][int(size*0.75):], test_dict['value'][int(size*0.75):], test_dict['label'][int(size*0.75):], 50, save_path='./save_online_model_afm')