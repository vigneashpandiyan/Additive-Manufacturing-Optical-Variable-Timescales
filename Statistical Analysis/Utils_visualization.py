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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import joypy
from matplotlib import cm
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


def kdeplot(data, feature, path_, label_custom):
    # Function to generate a KDE plot for a given feature in the data
    # Input: data (pandas DataFrame), feature (str), path_ (str), label_custom (list)
    # Output: fig (matplotlib Figure)

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
        fig = sns.kdeplot(data_1['Feature'], shade=True,
                          alpha=.5, color=c, label=label_custom[j])

    plt.title(feature)
    plt.legend()

    plot = feature+' kdeplot Distribution.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    return fig


def hist(data, feature, path_, label_custom):
    """
    This function generates a histogram plot for a given feature in the dataset.

    Parameters:
        data (pandas.DataFrame): The input dataset.
        feature (str): The name of the feature for which the histogram is to be generated.
        path_ (str): The path where the histogram plot will be saved.
        label_custom (list): A list of custom labels for the histogram plot.

    Returns:
        matplotlib.figure.Figure: The generated histogram plot.
    """
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
        fig = plt.hist(data_1['Feature'], alpha=0.5,
                       bins=50, color=c, label=label_custom[j])

    plt.title(feature)
    plt.legend()

    plot = feature+' hist Distribution.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    return fig


def violinplot(data, feature, path_):
    """
    Generate a violin plot to visualize the distribution of a feature across different target classes.

    Parameters:
    data (DataFrame): The input data containing the feature and target columns.
    feature (str): The name of the feature to be visualized.
    path_ (str): The path to save the generated plot.

    Returns:
    fig: The generated violin plot figure.
    """
    df = data
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))
    fig = plt.subplots(figsize=(10, 5))
    ax = sns.violinplot(x="target", y="Feature", data=df, palette=color)

    ax.set_title(str(feature) + ' distribution', fontsize=15)

    plt.legend(loc='best', frameon=False,
               fontsize=25, bbox_to_anchor=(1.10, 1.05))

    plt.xticks(rotation=45, fontsize=15)
    plt.xlabel('Layer thickness', fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Data spread', fontsize=15)

    plot = feature+' violinplot.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight', dpi=800)
    return fig


def boxplot(data, feature, path_):
    """
    Generate a boxplot visualization for a given feature in the dataset.

    Parameters:
    data (DataFrame): The input dataset.
    feature (str): The name of the feature to visualize.
    path_ (str): The path to save the generated plot.

    Returns:
    fig: The generated boxplot figure.
    """
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
    """
    Generate histogram and kernel density estimation plots for each target class in the given data.

    Args:
        data (pandas.DataFrame): The input data containing the target variable and feature.
        feature (str): The name of the feature to be plotted.
        path_ (str): The path to save the generated plot.

    Returns:
        matplotlib.figure.Figure: The generated figure object.

    """
    df = data

    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)

    uniq = np.sort(data.target)
    classes = np.unique(data.target)

    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    fig5, axes = plt.subplots(11, 1, figsize=(
        8, 16), dpi=800, sharex=True, sharey=True)

    # fig5, axes = plt.subplots(int(np.sqrt(len(classes))), int(
    #     np.sqrt(len(classes))), figsize=(10, 10), dpi=800, sharex=True, sharey=True)

    for i, (ax, target) in enumerate(zip(axes.flatten(), df.target.unique())):
        c = next(color)
        x = df.loc[df.target == target, 'Feature']
        ax.hist(x, alpha=0.7, bins=50, density=True,
                stacked=True, label=str(target), color=c)
        sns.kdeplot(x, shade=True, alpha=.5, color=c, label=str(target), ax=ax)
        ax.set_title(target)
        ax.legend()

    # plt.suptitle(feature, y=1.05, size=12)
    plt.title(feature)
    # ax.set_xlim(50, 70); ax.set_ylim(0, 1);

    # plt.tight_layout()

    plot = feature+' Histplotsplit.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    return fig5


def barplot(data, feature, path_):
    """
    Generate a bar plot for a given feature in the dataset.

    Parameters:
    data (DataFrame): The input dataset.
    feature (str): The name of the feature to plot.
    path_ (str): The path to save the generated plot.

    Returns:
    None
    """

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


def ridgeplot(data, feature, path_):
    """
    Generate a ridge plot visualization.

    Args:
        data (pandas.DataFrame): The input data.
        feature (str): The name of the feature to be visualized.
        path_ (str): The path to save the generated plot.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """

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
    """
    Generate KDE plots for each class in the given dataset.

    Args:
        data (pandas.DataFrame): The input dataset.
        feature (str): The name of the feature to plot.
        path_ (str): The path to save the generated plot.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    df = data
    # Rename the last column to 'target'
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns

    # Shuffle the data
    data = data.sample(frac=1.0)
    global_min = data['Feature'].min()
    global_max = data['Feature'].max()
    # Get unique classes
    classes = np.unique(data['target'])

    # Set up color iterator
    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    # Dynamically calculate the number of rows and columns for subplots
    n_classes = len(classes)
    n_cols = 2

    if n_classes//2 == 0:
        n_rows = (n_classes) // n_cols
    else:
        n_rows = (n_classes + 1) // n_cols  # Calculate the number of rows

    # Create a figure with GridSpec for better control
    fig5 = plt.figure(figsize=(10, 12), constrained_layout=True)
    # plt.subplots_adjust(wspace=-0.2, hspace=-0.2)
    gs = GridSpec(n_rows, n_cols, figure=fig5)
    # gs.update(wspace=0.5)
    axes = []

    # Generate the KDE plots
    for i, target in enumerate(df.target.unique()):

        if n_classes % n_cols != 0 and i == n_classes - 1:  # Center the last plot if odd
            # ax = fig5.add_subplot(gs[i // n_cols, :])  # Span both columns
            # Span both columns
            ax = plt.subplot2grid((n_rows, n_classes),
                                  (i//2, 2), colspan=6)
            pos = ax.get_position()  # Get the original position
            new_pos = [pos.x0 + 0.05, pos.y0 - 0.15, pos.width,
                       pos.height]  # Shift to the left by 0.05
            ax.set_position(new_pos)
            print("Loop 2")
            print(i // n_cols)

        else:
            ax = fig5.add_subplot(gs[i // n_cols, i % n_cols])

            print("Loop 1")
            print(i)

        ax.set_xlabel('Data spread', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.tick_params(axis='x', labelsize=16)  # Change font size for x-ticks
        ax.tick_params(axis='y', labelsize=16)  # Change font size for y-ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # For X-axis
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axes.append(ax)
        c = next(color)
        # Using the actual feature argument
        x = data.loc[data['target'] == target, 'Feature']
        sns.kdeplot(x, fill=True, alpha=.9, color=c, label=str(target), ax=ax)
        ax.set_title(f"Layer thickness: {target}", fontsize=16)
        ax.legend()
        # ax.tight_layout()
        ax.set_xlim(global_min, global_max)

    # Adjust layout and save the plot
    # plt.suptitle(str(feature) + ' distribution', fontsize=15)
    # plt.tight_layout()
    fig5.subplots_adjust(hspace=0.4, wspace=0.3)
    plot = feature + '_kdeplotsplit.png'
    plt.savefig(os.path.join(path_, plot), bbox_inches='tight')
    plt.show()
    return fig5
