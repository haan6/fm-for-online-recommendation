import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def draw_graph(filepath, filename):
    with open(filepath+filename, 'rb') as f:
        result_dict = pickle.load(f)

    xy_line = (0, 1)
    fig, ax = plt.subplots(figsize=(12, 9))

    roc_scores = result_dict['roc']
    ratio = result_dict['data_ratio']

    for i, (key, roc_score) in enumerate(roc_scores.items()):
        ax.plot([roc['fpr'] for roc in roc_score], [roc['tpr'] for roc in roc_score], marker='o', linestyle='', label=key)
        fpr, tpr = [roc['fpr'] for roc in roc_score], [roc['tpr'] for roc in roc_score]
        ax.annotate(str(ratio[key][int(len(fpr)/4*1)]), (fpr[int(len(fpr)/4*1)], tpr[int(len(tpr)/4*1)]))
        ax.annotate(str(ratio[key][int(len(fpr)/4*2)]), (fpr[int(len(fpr)/4*2)], tpr[int(len(tpr)/4*2)]))
        ax.annotate(str(ratio[key][int(len(fpr)/4*3)]), (fpr[int(len(fpr)/4*3)], tpr[int(len(tpr)/4*3)]))
        ax.annotate(str(ratio[key][int(len(fpr)/4*4)-1]), (fpr[int(len(fpr)/4*4)-1], tpr[int(len(tpr)/4*4)-1]))

    ax.plot(xy_line, 'r--', label='Random guess')

    # add labels, legend and make it nicer
    ax.set_xlabel('FPR or (1 - specificity)')
    ax.set_ylabel('TPR or sensitivity')
    ax.set_title('ROC Space')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    plt.savefig(f"performance/plot/{filename.split('.')[0]}_roc.png", dpi=100)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 9))
    acc_scores = result_dict['accuracy']
    ratio = result_dict['data_ratio']

    j = 0
    for key, acc_score in acc_scores.items():
        ax.plot([i for i in range(len(acc_score))], [acc * 0.01 for acc in acc_score], marker='o', linestyle='', label=key)
        for i in range(len(acc_score)):
            if j == 0:
                plt.xticks(np.arange(len(acc_score)),
                           [f"{ratio[key][i][0]} : {ratio[key][i][1]}" for i in range(len(acc_score))],
                           fontsize=7)
                j += 1

    ax.set_xlabel('Negative #: Positive #')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Scores')
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"performance/plot/{filename.split('.')[0]}_acc.png", dpi=100)
    plt.show()

if __name__ == "__main__":
    for file in os.listdir('performance/save_log/'):
        draw_graph("performance/save_log/", file)
