import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns


def xy_plot(x, y):
    '''
        Simple xy plot with seaborn, needs some styling improvments
    '''
    sns.lineplot(x, y)
 
def arg_max_plot(x, y):
    '''
        Highlights with a red cross the highest point in y, needs some styling improvments
    '''
    plt.plot(x[np.argmax(y)],np.max(y), 'rx')

def univariate_outlier_id_plot(df):
    df_num = df.select_dtypes(include=["number"]).drop(["Response"], axis = 1)
    color = "gray"
    fig = plt.figure(figsize=(8, 25))
    i=1
    for feature in df_num:
        if feature == "Income":
          ser = df_num[feature].copy()
          ser.dropna(inplace=True)
        else:
          ser = df_num[feature]
        ax = fig.add_subplot(df_num.shape[1], 2, i)
        box = ax.boxplot(ser, flierprops=dict(markerfacecolor='r', marker='s'), vert=False, patch_artist=True)
        box['boxes'][0].set_facecolor(color)
        ax.set_title("Boxplot of "+feature)
        ax = fig.add_subplot(df_num.shape[1], 2, i+1)
        ax.hist(ser, density=1, bins=30, color=color, alpha=0.7, rwidth=0.85)
        ax.set_title("Histogram of "+feature)
        i+=2

    plt.tight_layout()
    plt.show()


def filter_by_std(series_, n_stdev=3.0, return_thresholds=False):
    mean_, stdev_ = series_.mean(), series_.std()
    cutoff = stdev_ * n_stdev
    lower_bound, upper_bound = mean_ - cutoff, mean_ + cutoff
    if return_thresholds:
        return lower_bound, upper_bound
    else:
        return [True if i < lower_bound or i > upper_bound else False for i in series_]


def filter_by_iqr(series_, k=1.5, return_thresholds=False):
    q25, q75 = np.percentile(series_, 25), np.percentile(series_, 75)
    iqr = q75 - q25

    cutoff = iqr * k
    lower_bound, upper_bound = q25 - cutoff, q75 + cutoff

    if return_thresholds:
        return lower_bound, upper_bound
    else:
        return [True if i < lower_bound or i > upper_bound else False for i in series_]


def plot_filter_by_stdev(df, feature, stdev_tuple=(3.0, 2.0), colors=("red", "yellow")):
    df_num = df.select_dtypes(include=["number"]).drop(["Response"], axis=1)
    sns.distplot(df_num[feature], kde=False, color="gray")
    lower_bound_1, upper_bound_1 = filter_by_std(df_num[feature], n_stdev=stdev_tuple[0], return_thresholds=True)
    lower_bound_2, upper_bound_2 = filter_by_std(df_num[feature], n_stdev=stdev_tuple[1], return_thresholds=True)
    if df_num[feature].min() <= 0:
        plt.axvspan(min(df_num[feature][df_num[feature] < lower_bound_1], default=df_num[feature].min()), lower_bound_1, alpha=0.2,
                    color=colors[0])
        plt.axvspan(min(df_num[feature][df_num[feature] < lower_bound_2], default=df_num[feature].min()), lower_bound_1, alpha=0.2,
                    color=colors[1])
    plt.axvspan(upper_bound_1, max(df_num[feature][df_num[feature] > upper_bound_1], default=df_num[feature].max()), alpha=0.2,
                color=colors[0])
    plt.axvspan(upper_bound_2, max(df_num[feature][df_num[feature] > upper_bound_2], default=df_num[feature].max()), alpha=0.2,
                color=colors[1])
    plt.title("Outliers in {} by {} and {} standard deviations:\n".format(feature, stdev_tuple[0], stdev_tuple[1]))



def plot_filter_by_iqr(df, feature, k_tuple=(1.7, 1.2), colors=("red", "yellow")):
    df_num = df.select_dtypes(include=["number"]).drop(["Response"], axis=1)
    sns.distplot(df_num[feature], kde=False, color="gray")
    lower_bound_1, upper_bound_1 = filter_by_iqr(df_num[feature], k=k_tuple[0], return_thresholds=True)
    lower_bound_2, upper_bound_2 = filter_by_iqr(df_num[feature], k=k_tuple[1], return_thresholds=True)
    if df_num[feature].min() <= 0:
        plt.axvspan(min(df_num[feature][df_num[feature] < lower_bound_1], default=df_num[feature].min()), lower_bound_1, alpha=0.2,
                    color=colors[0])
        plt.axvspan(min(df_num[feature][df_num[feature] < lower_bound_2], default=df_num[feature].min()), lower_bound_1, alpha=0.2,
                    color=colors[1])
    plt.axvspan(upper_bound_1, max(df_num[feature][df_num[feature] > upper_bound_1], default=df_num[feature].max()), alpha=0.2,
                color=colors[0])
    plt.axvspan(upper_bound_2, max(df_num[feature][df_num[feature] > upper_bound_2], default=df_num[feature].max()), alpha=0.2,
                color=colors[1])
    plt.title("Outliers in {} by {} and {} k in IQR:\n".format(feature, k_tuple[0], k_tuple[1]))


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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
