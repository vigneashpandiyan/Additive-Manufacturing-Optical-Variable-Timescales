# -*- coding: utf-8 -*-
"""
@author: vpsora
contact: vigneashwara.solairajapandiyan@utu.fi,vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work
"Adaptive In-situ Monitoring for Laser Powder Bed Fusion:Self-Supervised Learning for Layer Thickness Monitoring Across Scan lengths based on Pyrometry"

@any reuse of this code should be authorized by the code author
"""
# %%
# Libraries to import
from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import itertools
from matplotlib import animation
import torch
import os

# %%


def tsne_visualization_(X, Y, folder_created, filename, epoch_length):
    """
    Visualizes the t-SNE lower dimensional embeddings computed on a window length of data points.

    Args:
        X (numpy.ndarray): The input data points.
        Y (numpy.ndarray): The labels corresponding to the input data points.
        folder_created (str): The path of the folder where the visualization files will be saved.
        filename (str): The name of the file for the visualization.
        epoch_length (int): The length of the epoch.

    Returns:
        tuple: A tuple containing the input data points (X), labels (Y), the axis object (ax), the figure object (fig),
               and the name of the graph file (graph_name).
    """
    graph_name1 = str(filename)+'_2D'+'_'+str(epoch_length)+'.png'
    graph_name1 = os.path.join(folder_created, graph_name1)

    graph_name2 = str(filename)+'_3D'+'_'+str(epoch_length)+'.png'
    graph_name2 = os.path.join(folder_created, graph_name2)

    graph_name3 = str(filename)+'_3D'+'_'+str(epoch_length)+'.gif'
    graph_name3 = os.path.join(folder_created, graph_name3)

    Save = str(filename) + '_TSNE'+'_'+str(epoch_length)+'.npy'
    Save = os.path.join(folder_created, Save)

    Save_Label = str(filename)+'_label_'+str(epoch_length)+'.npy'
    Save_Label = os.path.join(folder_created, Save_Label)

    print(Save_Label)

    title = 'T-sne lower dimensional embeddings computed' + "\n" + \
        'on window length of '+str(1000-epoch_length)+' data points'

    ax, fig, graph_name = TSNEplot_(X, Y, graph_name1, graph_name2,
                                    graph_name3, Save, str(title), limits=2.6, perplexity=20)

    np.save(Save_Label, Y)

    return X, Y, ax, fig, graph_name


def TSNEplot_(output, group, graph_name1, graph_name2, graph_name3, Save, graph_title, limits, perplexity):
    """
    Plots t-SNE embeddings in 2D and 3D.

    Args:
        output (numpy.ndarray): Array of latent space, features fed row-wise.
        group (numpy.ndarray): Array of target values.
        graph_name1 (str): Name of the 2D graph file to be saved.
        graph_name2 (str): Name of the 3D graph file to be saved.
        graph_name3 (str): Name of the graph file to be returned.
        Save (str): Path to save the t-SNE embeddings.
        graph_title (str): Title of the graph.
        limits (list): List of limits for the graph axes.
        perplexity (int): Perplexity value for t-SNE.

    Returns:
        ax (matplotlib.axes.Axes): Axes object of the 3D graph.
        fig (matplotlib.figure.Figure): Figure object of the 3D graph.
        graph_name3 (str): Name of the graph file.

    """
    # array of latent space, features fed rowise

    output = np.array(output)
    group = np.array(group)

    print('target shape: ', group.shape)
    print('output shape: ', output.shape)
    print('perplexity: ', perplexity)

    group = np.ravel(group)
    RS = np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)

    np.save(Save, tsne_fit)

    plot_2Dembeddings_(tsne_fit, group, graph_name1,
                       graph_title, limits, xlim=None, ylim=None)
    ax, fig = plot_3Dembeddings_(tsne_fit, group, graph_name2,
                                 graph_title, limits, xlim=None, ylim=None)

    return ax, fig, graph_name3


def plot_2Dembeddings_(tsne_fit, group, graph_name1, graph_title, limits, xlim=None, ylim=None):
    """
    Plots 2D embeddings using t-SNE algorithm.

    Args:
        tsne_fit (numpy.ndarray): The t-SNE fit array.
        group (numpy.ndarray): The group array.
        graph_name1 (str): The name of the output graph file.
        graph_title (str): The title of the graph.
        limits (int): The limit value for the graph.
        xlim (tuple, optional): The x-axis limits. Defaults to None.
        ylim (tuple, optional): The y-axis limits. Defaults to None.

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

    # marker= ["o","*",">","o","*",">"]

    # color = [ '#dd221c', '#16e30b', 'blue','#fab6b6','#a6ffaa','#b6befa']

    labels = ['10 um', '20 um', '30 um', '40 um',
              '50 um', '60 um', '70 um', '80 um', '90 um', '100 um', '110 um']

    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(9, 9), dpi=200)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)

    a1, a2 = detect_limits(x1, limits)
    b1, b2 = detect_limits(x2, limits)

    plt.ylim(b1, b2)
    plt.xlim(a1, a2)

    for i in range(len(uniq)):

        c = next(color)
        m = next(marker)
        indx = (df['label']) == uniq[i]
        a = x1[indx]
        b = x2[indx]
        plt.plot(a, b, color=c, label=labels[i], marker=m, linestyle='', ms=10)

    plt.xlabel('Dimension 1', labelpad=15, fontsize=25)
    plt.ylabel('Dimension 2', labelpad=15, fontsize=25)

    plt.title(graph_title, pad=15, fontsize=25)
    plt.legend()
    plt.locator_params(nbins=6)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1, 1), frameon=False)
    plt.savefig(graph_name1, bbox_inches='tight', dpi=200)
    plt.show()


def detect_limits(scores_normal, limits):
    """
    Detects the lower and upper limits based on the given scores and limits.

    Args:
        scores_normal (numpy.ndarray): An array of scores.
        limits (float): The number of standard deviations to consider for calculating the limits.

    Returns:
        tuple: A tuple containing the lower limit (Threshold0) and upper limit (Threshold1).
    """

    # find q1 and q3 values
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    Threshold0 = normal_avg - (normal_std * limits)
    Threshold1 = normal_avg + (normal_std * limits)
    return Threshold0, Threshold1


def plot_3Dembeddings_(tsne_fit, group, graph_name2, graph_title, limits, xlim=None, ylim=None):
    """
    Plots 3D embeddings using t-SNE algorithm.

    Args:
        tsne_fit (numpy.ndarray): The t-SNE fit array with shape (n_samples, 3).
        group (numpy.ndarray): The group labels for each sample with shape (n_samples,).
        graph_name2 (str): The name of the output graph file.
        graph_title (str): The title of the graph.
        limits (int): The limit value for detecting outliers.
        xlim (tuple, optional): The x-axis limits of the plot. Defaults to None.
        ylim (tuple, optional): The y-axis limits of the plot. Defaults to None.

    Returns:
        tuple: A tuple containing the matplotlib 3D axes object and the figure object.

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

    labels = ['10 um', '20 um', '30 um', '40 um',
              '50 um', '60 um', '70 um', '80 um', '90 um', '100 um', '110 um']

    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure(figsize=(15, 15), dpi=200)
    fig.set_facecolor('white')

    plt.rcParams["legend.markerscale"] = 2
    plt.rc("font", size=25)

    ax = plt.axes(projection='3d')
    ax.grid(False)
    ax.view_init(elev=15, azim=125)  # 115
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

    for i in range(len(uniq)):

        d = next(color)
        m = next(marker)

        print(i)
        indx = (df['label']) == uniq[i]
        a = x1[indx]
        b = x2[indx]
        c = x3[indx]
        ax.plot(a, b, c, color=d,
                label=labels[i], marker=m, linestyle='', ms=10)

    plt.xlabel('Dimension 1', labelpad=10, fontsize=20)
    plt.xticks(fontsize=20)

    plt.ylabel('Dimension 2', labelpad=10, fontsize=20)
    plt.yticks(fontsize=20)

    ax.set_zlabel('Dimension 3', labelpad=10, fontsize=20)
    ax.zaxis.set_tick_params(labelsize=20)

    ax.dist = 11

    plt.title(graph_title, fontsize=25)
    plt.locator_params(nbins=6)
    plt.legend(loc='upper left', frameon=False)
    plt.savefig(graph_name2, bbox_inches='tight', dpi=200)
    plt.show()

    return ax, fig
