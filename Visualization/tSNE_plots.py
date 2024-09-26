# -*- coding: utf-8 -*-
"""
@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch,vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"

@any reuse of this code should be authorized by the code author
"""
# %%
# Libraries to import

import itertools
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.manifold import TSNE
np.random.seed(1974)
plt.rcParams.update(plt.rcParamsDefault)
# marker= ["o","*",">","o","*",">"]
# color = [ '#dd221c', '#16e30b', 'blue','#fab6b6','#a6ffaa','#b6befa']


def plot_2Dembeddings(tsne_fit, group, graph_name1, graph_title, epoch_length, limits, xlim=None, ylim=None):
    """
    Plots 2D embeddings using t-SNE algorithm.

    Args:
        tsne_fit (numpy.ndarray): The t-SNE fit array with shape (n_samples, 2).
        group (numpy.ndarray): The group labels for each sample with shape (n_samples,).
        graph_name1 (str): The name of the output graph file.
        graph_title (str): The title of the graph.
        epoch_length (int): The length of the epoch.
        limits (float): The limit for detecting the range of the data.
        xlim (tuple, optional): The x-axis limits of the plot. Defaults to None.
        ylim (tuple, optional): The y-axis limits of the plot. Defaults to None.

    Returns:
        None
    """

    x1 = tsne_fit[:, 0]
    x2 = tsne_fit[:, 1]

    group = np.ravel(group)
    df = pd.DataFrame(dict(x=x1, y=x2, label=group))
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq = np.sort(uniq)

    print(uniq)

    classes = np.unique(uniq)
    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))
    marker = itertools.cycle(('X', '+', '.', 'o', '*', '>', 'D'))

    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(9, 9), dpi=200)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)

    a1, a2 = detect_limits(x1, limits)
    b1, b2 = detect_limits(x2, limits)

    plt.ylim(b1, b2)
    plt.xlim(a1, a2)

    labels = ['10 um', '20 um', '30 um', '40 um',
              '50 um', '60 um', '70 um', '80 um', '90 um', '100 um', '110 um']

    for i in range(len(uniq)):

        c = next(color)
        m = next(marker)
        indx = (df['label']) == uniq[i]
        a = x1[indx]
        b = x2[indx]
        plt.plot(a, b, color=c, label=labels[i], marker=m, linestyle='', ms=10)

    plt.xlabel('Dimension 1', labelpad=15, fontsize=25)
    plt.ylabel('Dimension 2', labelpad=15, fontsize=25)

    graph_title = 'T-sne lower dimensional embeddings computed' + "\n" + \
        'on window length of '+str(1000-epoch_length)+' data points'

    plt.title(graph_title, pad=15, fontsize=25)
    plt.legend()
    plt.locator_params(nbins=6)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1, 1), frameon=False)
    plt.savefig(graph_name1, bbox_inches='tight', dpi=200)
    plt.show()


def TSNEplot(output, group, graph_name1, graph_name2, graph_name3, graph_title, epoch_length, limits, perplexity):
    """
    This function generates t-SNE plots based on the given input data.

    Parameters:
    output (numpy.ndarray): The output data for t-SNE plot.
    group (numpy.ndarray): The group data for t-SNE plot.
    graph_name1 (str): The name of the first graph.
    graph_name2 (str): The name of the second graph.
    graph_name3 (str): The name of the third graph.
    graph_title (str): The title of the graph.
    epoch_length (int): The length of the epoch.
    limits (int): The limits for the graph.
    perplexity (float): The perplexity value for t-SNE.

    Returns:
    ax (matplotlib.axes.Axes): The axes object of the third graph.
    fig (matplotlib.figure.Figure): The figure object of the third graph.
    graph_name3 (str): The name of the third graph.
    """
    output = np.array(output)
    group = np.array(group)

    print('target shape: ', group.shape)
    print('output shape: ', output.shape)
    print('perplexity: ', perplexity)

    group = np.ravel(group)
    RS = np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)

    plot_2Dembeddings(tsne_fit, group, graph_name1, graph_title,
                      epoch_length, limits, xlim=None, ylim=None)
    ax, fig = plot_3Dembeddings(tsne_fit, group, graph_name2,
                                graph_title, epoch_length, limits, xlim=None, ylim=None)

    return ax, fig, graph_name3


def plot_3Dembeddings(tsne_fit, group, graph_name2, graph_title, epoch_length, limits, xlim=None, ylim=None):
    """
    Plots 3D embeddings using t-SNE algorithm.

    Args:
        tsne_fit (numpy.ndarray): The t-SNE fit array with shape (n_samples, 3).
        group (numpy.ndarray): The group labels for each sample with shape (n_samples,).
        graph_name2 (str): The name of the output graph file.
        graph_title (str): The title of the graph.
        epoch_length (int): The length of the epoch.
        limits (int): The limit value.
        xlim (tuple, optional): The x-axis limits. Defaults to None.
        ylim (tuple, optional): The y-axis limits. Defaults to None.

    Returns:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes object.
        fig (matplotlib.figure.Figure): The figure object.
    """

    x1 = tsne_fit[:, 0]
    x2 = tsne_fit[:, 1]
    x3 = tsne_fit[:, 2]

    group = np.ravel(group)

    df = pd.DataFrame(dict(x=x1, y=x2, z=x3, label=group))

    groups = df.groupby('label')
    uniq = list(set(df['label']))
    print(uniq)
    uniq = np.sort(uniq)

    classes = np.unique(uniq)
    print("classes.............", len(classes))
    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    marker = itertools.cycle(('X', '+', '.', 'o', '*', '>', 'D'))

    plt.rcParams.update(plt.rcParamsDefault)

    # fig = plt.figure(figsize=(15, 15), dpi=200)
    fig = plt.figure(constrained_layout=True)
    fig.set_facecolor('white')

    plt.rcParams["legend.markerscale"] = 1
    # plt.rc("font", size=35)

    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    ax.grid(False)
    ax.view_init(elev=15, azim=125)  # 115
    # ax.view_init(elev=30, azim=30)
    ax.set_facecolor('white')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    a1, a2 = detect_limits(x1, limits)
    b1, b2 = detect_limits(x2, limits)
    c1, c2 = detect_limits(x3, limits)

    ax.set_ylim(b1, b2)
    ax.set_zlim(c1, c2)
    ax.set_xlim(a1, a2)

    labels = ['10 um', '20 um', '30 um', '40 um',
              '50 um', '60 um', '70 um', '80 um', '90 um', '100 um', '110 um']

    for i in range(len(uniq)):

        d = next(color)
        m = next(marker)

        print(i)
        indx = (df['label']) == uniq[i]
        a = x1[indx]
        b = x2[indx]
        c = x3[indx]
        ax.plot(a, b, c, color=d,
                label=labels[i], marker=m, linestyle='')  # ms=10

    # plt.xlabel('Dimension 1', labelpad=15, fontsize=10)
    plt.xlabel('Dimension 1',  fontsize=10)
    # plt.xticks(fontsize=20)

    # plt.ylabel('Dimension 2', labelpad=15, fontsize=10)
    plt.ylabel('Dimension 2', fontsize=10)
    # plt.yticks(fontsize=20)

    # ax.set_zlabel('Dimension 3', labelpad=10, fontsize=10)
    ax.set_zlabel('Dimension 3', fontsize=10)
    # ax.zaxis.set_tick_params(labelsize=25)
    # ax.dist = 20
    graph_name = 'T-sne lower dimensional embeddings computed' + "\n" + \
        'on window length of '+str(1000-epoch_length)+' data points'

    plt.title(graph_name, fontsize=10)
    plt.locator_params(nbins=6)
    plt.legend(loc='upper left', frameon=False)
    plt.savefig(graph_name2, bbox_inches='tight', dpi=200)
    # plt.savefig(graph_name2, dpi=200)
    plt.show()

    return ax, fig

# %%


def detect_limits(scores_normal, limits):
    """
    Detects the lower and upper limits based on the given scores and limits.

    Args:
        scores_normal (numpy.ndarray): An array of scores.
        limits (float): The number of standard deviations to consider for the limits.

    Returns:
        tuple: A tuple containing the lower limit (Threshold0) and upper limit (Threshold1).
    """

    # find q1 and q3 values
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    Threshold0 = normal_avg - (normal_std * limits)
    Threshold1 = normal_avg + (normal_std * limits)
    return Threshold0, Threshold1
