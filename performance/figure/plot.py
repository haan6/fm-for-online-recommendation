import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_scores(filename):
    with open(f'performance/{filename}.pickle', 'wb') as f:
         result = pickle.load(f)

    print("===== Drawing Accuracy Plot =====")
    fig, ax = plt.subplots()
    plt.ylim(-4, 104)
    colors = ['r', 'g', 'b']

    for i, key in enumerate(result['model'].keys()):
        plt.plot([j for j in range(len(result['accuracy_scores'][key]))], result['accuracy_scores'][key],
                 color=colors[i], label=key)

    plt.title('Accuracy Score')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=1, fancybox=True, shadow=True)
    plt.savefig(f'performance/figure/{filename}_accuracy_score.png')
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

    mark = ['s', 'o', 'v']
    color = ['r', 'g', 'b']

    for i, key in enumerate(result['model'].keys()):
        tpr = []
        fpr = []

        for j in range(len(result['roc_scores'][key])):
            tpr.append([result['roc_scores'][key][j]["tpr"]])
            fpr.append([result['roc_scores'][key][j]["fpr"]])

        plt.plot(fpr, tpr, color=color,
                 marker=mark,
                 markerfacecolor='None',
                 markeredgecolor=color,
                 linestyle='None',
                 label=key)

    plt.title('ROC Score')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=1, fancybox=True, shadow=True)
    plt.savefig(f'performance/figure/{filename}_accuracy_score.png')
    plt.show()