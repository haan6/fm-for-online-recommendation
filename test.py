import torch
from model.model_instance import _make_models
from data import data_preprocess

import argparse
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np

########################################################################################################################
# Add Argument
########################################################################################################################

parser = argparse.ArgumentParser(description='Put hyperparameters')
parser.add_argument('--exp_num', nargs='?')
parser.add_argument('--dataset_name', nargs='?')
parser.add_argument('--option', nargs='?')
parser.add_argument('--max_num_hidden_layers', type=int)
parser.add_argument('--qtd_neuron_per_hidden_layer', type=int)
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--n_classes', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--n', type=float)
parser.add_argument('--version', nargs='?')
args = parser.parse_args()

exp_num = args.exp_num
dataset_name = args.dataset_name
option = args.option
max_num_hidden_layers = args.max_num_hidden_layers
qtd_neuron_per_hidden_layer = args.qtd_neuron_per_hidden_layer
embedding_size = args.embedding_size
n_classes = args.n_classes
batch_size = args.batch_size
n = args.n
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
                + '_NClasses' + str(n_classes) \
                + '_Version' + version \
                + '_Date' + date
print("======Save Path Done======")


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


with torch.cuda.device(0):
    model_names = ['ONN_NFM', 'FM', 'SGD_NFM']
    models = []
    time_elapsed = dict()
    accuracy_scores = dict()
    roc_scores = dict()

    print("===== Instantiating Models =====")
    for name in model_names:
        models.append((_make_models(field_size,
                                    train_dict['feature_sizes'],
                                    max_num_hidden_layers,
                                    qtd_neuron_per_hidden_layer,
                                    dropout_shallow=[0.5],
                                    embedding_size=embedding_size,
                                    n_classes=2,
                                    batch_size=batch_size,
                                    verbose=False,
                                    interaction_type=True,
                                    eval_metric=roc_auc_score,
                                    b=0.99,
                                    n=0.01,
                                    s=0.2,
                                    use_cuda=True,
                                    model_name=name)))
    print("===== Models Ready =====")

    for i, (model, option_name) in enumerate(models):
        print(f"======Training {option_name}======")
        time_elapsed[option_name], accuracy_scores[option_name], roc_scores[option_name] = model.evaluate(train_Xi, train_Xv, train_Y)
        print(f"======Evaluating {option_name} Done. Time Elapsed: {int(time_elapsed[option_name] / 60)}m {time_elapsed[option_name] - 60 * int(time_elapsed[option_name] / 60)}s======")
        print("roc", roc_scores[option_name][-1])
        print("acc", accuracy_scores[option_name][-1])

        with open(f'performance/{option_name}.pickle', 'wb') as f:
            pickle.dump(model, f)

    print("======Training Models Done======")

    print("===== Drawing Accuracy Plot =====")
    plt.ylim(-4, 104)
    colors = ['r', 'g', 'b']

    for i, color in enumerate(colors):
        plt.plot([j for j in range(len(accuracy_scores[models[i][1]]))], accuracy_scores[models[i][1]],
                 color=color, label=models[i][1])

    plt.title('Accuracy Score')
    plt.legend(loc='lower right')
    plt.savefig(f'{save_figure_path}{save_filename}_accuracy_score.png')
    plt.grid()

    plt.clf()
    print("===== Drawing ROC Plot =====")

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    x = np.linspace(*ax.get_xlim())
    plt.plot(x, x, color='black')

    plt.ylim(-0.04, 1.04)
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    for i, (mark, color) in enumerate(zip(
            ['s', 'o', 'v'], ['r', 'g', 'b'])):
        tpr = []
        fpr = []

        for j in range(len(roc_scores[models[i][1]])):
            tpr.append([roc_scores[models[i][1]][j]["tpr"]])
            fpr.append([roc_scores[models[i][1]][j]["fpr"]])

        plt.plot(fpr, tpr, color=color,
                 marker=mark,
                 markerfacecolor='None',
                 markeredgecolor=color,
                 linestyle='None',
                 label=models[i][1])

    plt.title('ROC Score')
    plt.legend(loc='lower right')
    plt.savefig(f'{save_figure_path}{save_filename}_roc_score.png')

    plt.show()
