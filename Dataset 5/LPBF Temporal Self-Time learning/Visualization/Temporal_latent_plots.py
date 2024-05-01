# -*- coding: utf-8 -*-
"""
@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch,vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"

@any reuse of this code should be authorized by the code author
"""
#%%
#Libraries to import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import metrics
import seaborn as sns
from matplotlib.pyplot import cm
import itertools
import seaborn as sns


def latent_plots(folder_created, graph_name, epoch_length):
    """
    Generate latent plots for visualization.

    Args:
        folder_created (str): Path to the folder where the embeddings and labels are stored.
        graph_name (str): Name of the graph.
        epoch_length (int): Length of each epoch.

    Returns:
        None
    """

    train_embeddings = folder_created + '/' + str(graph_name) + '_' + str(epoch_length) + '_embeddings' + '.npy'
    train_labelsname = folder_created + '/' + str(graph_name) + '_' + str(epoch_length) + '_labels' + '.npy'

    X_train = np.load(train_embeddings).astype(np.float64)
    y_train = np.load(train_labelsname).astype(np.float64)

    Featurespace = X_train
    classspace = y_train

    columns = np.atleast_2d(Featurespace).shape[1]
    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']

    print(columns)

    for i in range(columns):
        print(i)
        Featurespace_1 = Featurespace.transpose()
        data = (Featurespace_1[i])
        data = data.astype(np.float64)
        #data= abs(data)
        df1 = pd.DataFrame(data)
        df1.rename(columns={df1.columns[0]: "Feature"}, inplace=True)
        df2.rename(columns={df2.columns[0]: "categorical"}, inplace=True)
        data = pd.concat([df1, df2], axis=1)
        minval = min(data.categorical.value_counts())

        print(minval)
        data = pd.concat([data[data.categorical == cat].head(minval)
                         for cat in data.categorical.unique()])
        distribution_plot(data, i, folder_created)

    all_plot(X_train, y_train, folder_created)

def plot_windows(epochs, window_size, path):

    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(3, 5))
    window_length = [window_size - x for x in epochs]
    sns.displot(window_length, color='blue', edgecolors="k", kde=True,
                line_kws={'color': 'crimson', 'lw': 5, 'ls': ':'})
    plt.ylabel("Relative frequency")
    plt.xlabel("Window lengths")
    plt.title(len(epochs))
    plt.rc("font", size=12)
    plt.tight_layout()
    plot = path+'Window_length_Histplots.jpg'
    plt.savefig(plot, bbox_inches='tight', dpi=800)
    plt.show()


def distribution_plot(data, i, folder_created):
    """
    Plots the distribution of a feature in the given dataset.

    Args:
        data (pandas.DataFrame): The dataset containing the feature to be plotted.
        i (int): The index of the weight.
        folder_created (str): The path of the folder where the plot will be saved.

    Returns:
        None
    """


    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    print(uniq)

    classes = np.unique(uniq)
    print(classes)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    sns.set(style="white")

    fig = plt.subplots(figsize=(5, 3), dpi=800)

    for j in range(len(classes)):
        print(j)

        data_1 = data[data.target == j]
        data_1 = data_1.drop(labels='target', axis=1)
        c = next(color)
        fig = sns.kdeplot(data_1['Feature'], shade=True, alpha=.5, color=c)

    plt.title("Weight " + str(i+1))
    plt.legend(labels=classes, bbox_to_anchor=(1.49, 1.05))
    title = folder_created+'/'+'Dimension'+'_'+str(i+1)+'_'+'distribution_plot'+'.png'
    # plt.xlim([0.0, np.max(data)])
    # plt.ylim([0.0, 65])
    plt.xlabel('Weight distribution')
    plt.savefig(title, bbox_inches='tight')
    plt.show()

# %%

def Cummulative_plots(Featurespace, classspace, i, ax):

    

    columns = np.atleast_2d(Featurespace).shape[1]
    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']

    print(i)

    Featurespace_1 = Featurespace.transpose()
    data = (Featurespace_1[i])
    data = data.astype(np.float64)

    df1 = pd.DataFrame(data)
    df1.rename(columns={df1.columns[0]: "Feature"}, inplace=True)
    df2.rename(columns={df2.columns[0]: "categorical"}, inplace=True)
    data = pd.concat([df1, df2], axis=1)
    minval = min(data.categorical.value_counts())
    data = pd.concat([data[data.categorical == cat].head(minval)
                     for cat in data.categorical.unique()])

    Cummulative_dist_plot(data, i, ax)


def Cummulative_dist_plot(data, i, ax):
    """
    Plots the cumulative distribution of a given dataset.

    Args:
        data (DataFrame): The input dataset.
        i (int): The weight index.
        ax (AxesSubplot): The matplotlib axes object to plot on.

    Returns:
        None
    """

    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    print(uniq)

    classes = np.unique(uniq)
    print(classes)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    sns.set(style="white")

    ax.plot(figsize=(5, 5), dpi=800)

    for j in range(len(classes)):
        print(j)

        data_1 = data[data.target == j]
        data_1 = data_1.drop(labels='target', axis=1)
        c = next(color)
        sns.kdeplot(data_1['Feature'], shade=True, alpha=.5, color=c, ax=ax)

    
    ax.set_title("Weight " + str(i+1), y=1.0, pad=-14)
    ax.set_xlabel('Weight distribution')
    # ax.set_ylabel('Density')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def all_plot(X_train, y_train, folder_created):
    """
    Plots the temporal latent plots for each feature in the given dataset.

    Args:
        X_train (numpy.ndarray): The input training data.
        y_train (numpy.ndarray): The target training data.
        folder_created (str): The folder path where the plots will be saved.

    Returns:
        None
    """
    columns = X_train.shape[1]
    columns = columns/8
    fig, axs = plt.subplots(
        nrows=8,
        ncols=int(columns),
        sharey=False,
        figsize=(20, 20),
        dpi=600
    )

    columns = np.atleast_2d(X_train).shape[1]
    graph_name = folder_created+'/'+'Byol_Latent_'+str(columns)+'D_'+'.png'

    for i in range(columns):
        ax = axs.flat[i]
        Cummulative_plots(X_train, y_train, i, ax)

    fig.tight_layout()
    fig.savefig(graph_name)
    fig.show()
