# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch
The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"
@any reuse of this code should be authorized by the code author
"""

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

def plot_2Dembeddings(tsne_fit, group, graph_name1, graph_title,epoch_length, limits, xlim=None, ylim=None):

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

    for i in range(len(uniq)):

        c = next(color)
        m = next(marker)
        indx = (df['label']) == uniq[i]
        a = x1[indx]
        b = x2[indx]
        plt.plot(a, b, color=c, label=uniq[i], marker=m, linestyle='', ms=10)

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


def TSNEplot(output, group, graph_name1, graph_name2, graph_name3, graph_title,epoch_length, limits, perplexity):

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

    plot_2Dembeddings(tsne_fit, group, graph_name1, graph_title,epoch_length, limits, xlim=None, ylim=None)
    ax, fig = plot_3Dembeddings(tsne_fit, group, graph_name2,
                                graph_title,epoch_length, limits, xlim=None, ylim=None)

    return ax, fig, graph_name3


def plot_3Dembeddings(tsne_fit, group, graph_name2, graph_title,epoch_length, limits, xlim=None, ylim=None):

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

    fig = plt.figure(figsize=(15, 15), dpi=200)
    fig.set_facecolor('white')

    plt.rcParams["legend.markerscale"] = 2
    plt.rc("font", size=35)

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
        ax.plot(a, b, c, color=d, label=uniq[i], marker=m, linestyle='', ms=10)

    plt.xlabel('Dimension 1', labelpad=15, fontsize=20)
    plt.xticks(fontsize=20)

    plt.ylabel('Dimension 2', labelpad=15, fontsize=20)
    plt.yticks(fontsize=20)

    ax.set_zlabel('Dimension 3', labelpad=10, fontsize=20)
    ax.zaxis.set_tick_params(labelsize=25)
    ax.dist = 20
    graph_name = 'T-sne lower dimensional embeddings computed' + "\n" + \
        'on window length of '+str(1000-epoch_length)+' data points'
    
    plt.title(graph_name, fontsize=30)
    plt.locator_params(nbins=6)
    plt.legend(loc='upper left', frameon=False)
    plt.savefig(graph_name2, bbox_inches='tight', dpi=200)
    plt.show()

    return ax, fig


# %%
def detect_limits(scores_normal, limits):
    # find q1 and q3 values
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    Threshold0 = normal_avg - (normal_std * limits)
    Threshold1 = normal_avg + (normal_std * limits)
    return Threshold0, Threshold1
