# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch
The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"
@any reuse of this code should be authorized by the code author
"""
# %%
# Libraries to import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import joypy
from matplotlib import cm
from matplotlib import colors


def kdeplot(data, feature, path_, label_custom):

    fig = plt.subplots(figsize=(9, 5), dpi=800)
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))
    sns.set(style="white")

    for j in range(len(classes)):

        print(j)

        data_1 = data[data.target == j]
        data_1 = data_1.drop(labels='target', axis=1)
        c = next(color)
        fig = sns.kdeplot(data_1['Feature'], shade=True, alpha=.5, color=c, label=label_custom[j])

    plt.title(feature)
    plt.legend()

    plot = feature+' kdeplot Distribution.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    return fig


def hist(data, feature, path_, label_custom):

    fig = plt.subplots(figsize=(9, 5), dpi=800)
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))
    sns.set(style="white")

    for j in range(len(classes)):

        data_1 = data[data.target == j]
        data_1 = data_1.drop(labels='target', axis=1)
        c = next(color)
        fig = plt.hist(data_1['Feature'], alpha=0.5, bins=50, color=c, label=label_custom[j])

    plt.title(feature)
    plt.legend()

    plot = feature+' hist Distribution.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    return fig


def violinplot(data, feature, path_):
    df = data
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))
    fig = plt.subplots(figsize=(10, 5), dpi=800)
    ax = sns.violinplot(x="target", y="Feature", data=df, palette=color)

    ax.set_title(feature)

    plt.legend(loc='best', frameon=False,
               fontsize=20, bbox_to_anchor=(1.10, 1.05))

    plt.xticks(rotation=45)
    plt.xlabel('Layer thickness')
    plot = feature+' violinplot.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    return fig


def boxplot(data, feature, path_):
    df = data
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    fig = plt.subplots(figsize=(10, 5), dpi=800)
    ax = sns.boxplot(x="target", y="Feature", data=df, palette=color)
    plt.title(feature)
    plt.xticks(rotation=45)
    plt.legend(loc='lower right', frameon=False, fontsize=40)
    plt.xlabel('Layer thickness')
    plot = feature+' boxplot.png'
    plt.savefig(os.path.join(path_, plot),  bbox_inches='tight')
    return fig


def Histplotsplit(data, feature, path_):
    df = data

    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    fig5, axes = plt.subplots(11, 1, figsize=(8, 16), dpi=800, sharex=True, sharey=True)

    # fig5, axes = plt.subplots(int(np.sqrt(len(classes))), int(
    #     np.sqrt(len(classes))), figsize=(10, 10), dpi=800, sharex=True, sharey=True)

    for i, (ax, target) in enumerate(zip(axes.flatten(), df.target.unique())):
        c = next(color)
        x = df.loc[df.target == target, 'Feature']
        ax.hist(x, alpha=0.7, bins=50, density=True, stacked=True, label=str(target), color=c)
        sns.kdeplot(x, shade=True, alpha=.5, color=c, label=str(target), ax=ax)
        ax.set_title(target)
        ax.legend()

    # plt.suptitle(feature, y=1.05, size=12)
    plt.title(feature)
    #ax.set_xlim(50, 70); ax.set_ylim(0, 1);

    # plt.tight_layout()

    plot = feature+' Histplotsplit.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    return fig5


def barplot(data, feature, path_):
    df = data
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    fig6 = plt.subplots(figsize=(10, 5), dpi=800)
    ax = sns.barplot(x="target", y="Feature", data=df, palette=color, errcolor='gray', errwidth=2,
                     ci='sd')
    plt.title(str(feature))
    plt.xticks(rotation=45)
    plt.xlabel('Layer thickness')
    plot = feature+' Barplot.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    plt.show()


# %%https://github.com/sbebo/joypy/blob/master/Joyplot.ipynb

def ridgeplot(data, feature, path_):
    df = data

    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    # color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))
    plt.figure(dpi=800)
    fig6, axes = joypy.joyplot(data, by="target", figsize=(10, 17.5), overlap=4, bins=10, linecolor="black",
                               legend=False, colormap=colors.ListedColormap(cm.rainbow(np.linspace(0, 1, len(classes)))))
    plt.rc("font", size=20)
    # plt.title(feature)
    plot = feature+' Ridge.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    plt.show
    return fig6


def kdeplotsplit(data, feature, path_):
    df = data

    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    # fig5, axes = plt.subplots(int(np.sqrt(len(classes))), int(
    #     np.sqrt(len(classes))), figsize=(10, 10), dpi=800, sharex=True, sharey=True)

    fig5, axes = plt.subplots(11, 1, figsize=(8, 16), dpi=800, sharex=True, sharey=True)

    for i, (ax, target) in enumerate(zip(axes.flatten(), df.target.unique())):
        c = next(color)
        x = df.loc[df.target == target, 'Feature']
        sns.kdeplot(x, shade=True, alpha=.9, color=c, label=str(target), ax=ax)
        # ax.hist(x, alpha=0.5, bins=50, density=True, stacked=True, label=str(target), color=c)
        ax.set_title(target)
        ax.legend()

    # plt.title(str(feature))
    plt.tight_layout()
    plot = feature+' kdeplotsplit.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    return fig5
