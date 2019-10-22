import torch
from model.FM import *
from model.SGD_NFM import *
from model.ONN_NFM import *
from data import data_preprocess

import datetime
import matplotlib.pyplot as plt

print("===== Importing Dataset =====")

train_dict = data_preprocess.balance_criteo_data('data/tiny_train_input.csv', 'data/category_emb.csv')
train_dict_size = train_dict['size']

train_Xi, train_Xv, train_Y \
    = train_dict['index'][:int(train_dict_size * 0.05)], \
      train_dict['value'][:int(train_dict_size * 0.05)], \
      train_dict['label'][:int(train_dict_size * 0.05)]

print("===== Dataset is Ready =====")

with torch.cuda.device(0):
    time_elapsed = {"FM": 0, "SGD_NFM": 0, "ONN_NFM": 0}
    roc_scores = {"FM": [], "SGD_NFM": [], "ONN_NFM": []}

    print("===== Instantiating Models =====")

    fm = FM(39, train_dict['feature_sizes'], batch_size=20)
    sgd_nfm = SGD_NFM(39, train_dict['feature_sizes'], 5, 10, batch_size=20)
    onn_nfm = ONN_NFM(39, train_dict['feature_sizes'], max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=10,
                      verbose=True, use_cuda=True, interaction_type=True)

    models = [(fm, "FM"), (sgd_nfm, "SGD_NFM"), (onn_nfm, "ONN_NFM")]

    print("===== Models are Ready =====")

    for i, model in enumerate(models):
        print(f"===== Training {models[i][1]} =====")
        time_elapsed[models[i][1]], roc_scores[models[i][1]] = models[0].evaluate(train_Xi, train_Xv, train_Y)
        print(time_elapsed[models[i][1]], roc_scores[models[i][1]])
        print(f"===== Evaluating {models[i][1]} is Finished. Time: {time_elapsed[models[i][1]]} =====")
    #

    print("===== Drawing a plot =====")

    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d')

    fig, ax = plt.subplots()

    ax.ylim(-0.04, 1.04)
    ax.xlabel('FPR')
    ax.ylabel('TPR')

    for i, (mark, color) in enumerate(zip(
            ['s', 'o', 'v'], ['r', 'g', 'b'])):
        tpr = []
        fpr = []

        for j in range(roc_scores[models[i][1]]):
            tpr.append([roc_scores[models[i][1]][j]["tpr"]])
            fpr.append([roc_scores[models[i][1]][j]["fpr"]])
            ax.plot(fpr, tpr, color=color,
                    marker=mark,
                    markerfacecolor='None',
                    markeredgecolor=color,
                    linestyle='None',
                    label=models[i][0])

    ax.set_aspect('equal')
    ax.title('ROC Score')
    ax.legend(loc='lower right')
    fig.savefig(f'{date}_roc_score.png')

    plt.show()