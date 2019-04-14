import pandas as pd
import matplotlib.pyplot as plt

def plot_precision_recall_curve(recall, precision, auc):
    '''
        Credits to professor Ilya
    '''
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, marker='.', label=" (AUPR (unseen) {:.2f}".format(auc) + ")")
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.xlabel('Recall (unseen)')
    plt.ylabel('Precision (unseen)')
    plt.title('PR curve on unseen data')
    plt.legend(loc='best', title="Models")
    plt.show()  