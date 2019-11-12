import torch
from model.model_instance import _make_models
from data import data_preprocess

import argparse
import datetime
from sklearn.metrics import roc_auc_score
import pickle
from performance.figure import plot


########################################################################################################################
# Add Argument
########################################################################################################################

parser = argparse.ArgumentParser(description='Put hyperparameters')
parser.add_argument('--exp_num', nargs='?')
parser.add_argument('--dataset_name', nargs='?')
parser.add_argument('--option', nargs='?')
parser.add_argument('--version', nargs='?')
args = parser.parse_args()

exp_num = args.exp_num
dataset_name = args.dataset_name
option = args.option
version = args.version


########################################################################################################################
# Save path
########################################################################################################################
print("======Save Path======")
now = datetime.datetime.now()
date = now.strftime('%Y-%m-%d')

save_performance_path = 'performance/'
save_figure_path = 'performance/figure/'
save_filename = str(exp_num) \
                + '_Dataset' + dataset_name \
                + '_' + str(option) \
                + '_' + version \
                + '_Date' + date
print("======Save Path Done======")

if option == "Option1":
########################################################################################################################
# Option 1
########################################################################################################################
    max_num_hidden_layers_list = [1, 5, 10, 15, 20]
    qtd_neuron_per_hidden_layer_list = [10, 10, 10, 10, 10]
    embedding_size_list = [4, 4, 4, 4, 4]
    n_list = [0.01, 0.01, 0.01, 0.01, 0.01]

elif option == "Option2":
########################################################################################################################
# Option 2
########################################################################################################################
    max_num_hidden_layers_list = [5, 5, 5, 5, 5]
    qtd_neuron_per_hidden_layer_list = [1, 5, 10, 15, 20]
    embedding_size = [4, 4, 4, 4, 4]
    n = [0.01, 0.01, 0.01, 0.01, 0.01]

elif option == "Option3":
########################################################################################################################
# Option 3
########################################################################################################################
    max_num_hidden_layers_list = [5, 5, 5, 5, 5]
    qtd_neuron_per_hidden_layer_list = [10, 10, 10, 10, 10]
    embedding_size = [1, 2, 4, 8, 16]
    n = [0.01, 0.01, 0.01, 0.01, 0.01]

elif option == "Option4":
########################################################################################################################
# Option 4
########################################################################################################################
    max_num_hidden_layers_list = [5, 5, 5, 5, 5]
    qtd_neuron_per_hidden_layer_list = [10, 10, 10, 10, 10]
    embedding_size = [4, 4, 4, 4, 4]
    n = [0.1, 0.05, 0.01, 0.005, 0.001]


# elif option == "Option5":
# ########################################################################################################################
# # Option 5
# ########################################################################################################################
else:
    max_num_hidden_layers_list = [5]
    qtd_neuron_per_hidden_layer_list = [10]
    embedding_size_list = [4]
    n_list = [0.01]


########################################################################################################################
# Importing Dataset
########################################################################################################################
print("===== Importing Dataset =====")
if dataset_name == 'criteo':
    train_dict = data_preprocess.balance_criteo_data('data/criteo/tiny_train_input.csv', 'data/criteo/category_emb.csv')
    train_dict_size = int(train_dict['size'] * 0.2)
elif dataset_name == 'cod-rna2':
    train_dict = data_preprocess.read_svm_file('data/cod-rna2/cod-rna2.scale')
    train_dict_size = train_dict['size']
else:
    train_dict = data_preprocess.balance_criteo_data('data/cod-rna2/tiny_train_input.csv', 'data/cod-rna2/category_emb.csv')
    train_dict_size = int(train_dict['size'] * 0.2)

save_filename += '_DataNum' + str(train_dict_size)
field_size = len(train_dict['feature_sizes'])

train_Xi, train_Xv, train_Y \
    = train_dict['index'][:int(train_dict_size)], \
      train_dict['value'][:int(train_dict_size)], \
      train_dict['label'][:int(train_dict_size)]

print(f"===== Dataset Ready -- # of Data: {int(train_dict_size)} -- =====")


########################################################################################################################
# Instantiate Models
########################################################################################################################
with torch.cuda.device(0):
    result = {'model': dict(), 'time_elapsed': dict(), 'accuracy_scores': dict(), 'roc_scores': dict()}

    print("===== Instantiating Models =====")
    for i, (max_num_hidden_layers, qtd_neuron_per_hidden_layer, embedding_size, n) \
            in enumerate(zip(max_num_hidden_layers_list, qtd_neuron_per_hidden_layer_list, embedding_size_list, n_list)):
        instance, model_name = _make_models(field_size,
                                         train_dict['feature_sizes'],
                                         max_num_hidden_layers,
                                         qtd_neuron_per_hidden_layer,
                                         dropout_shallow=[0.5],
                                         embedding_size=embedding_size,
                                         n_classes=2,
                                         batch_size=1,
                                         verbose=False,
                                         interaction_type=True,
                                         eval_metric=roc_auc_score,
                                         b=0.99,
                                         n=n,
                                         s=0.2,
                                         use_cuda=True,
                                         model_name='ONN_NFM')
        result['model'][model_name] = instance
    print("===== Models Ready =====")

    for i, (option_name, model) in enumerate(result['model'].items()):
        print(f"======Training {option_name}======")
        result['time_elapsed'][option_name], result['accuracy_scores'][option_name], result['roc_scores'][option_name] = model.evaluate(train_Xi, train_Xv, train_Y)
        print(f"======Evaluating {option_name} Done. Time Elapsed: {int(result['time_elapsed'][option_name] / 60)}m {result['time_elapsed'][option_name] % 60}s======")
        print("roc", result['roc_scores'][option_name][-1])
        print("acc", result['accuracy_scores'][option_name][-1])

        with open(f'performance/{option_name}.pickle', 'wb') as f:
            pickle.dump(model, f)

    print("======Training Models Done======")
    with open(f'performance/{save_filename}.pickle', 'wb') as f:
        pickle.dump(result, f)

    plot.plot_scores(save_filename)