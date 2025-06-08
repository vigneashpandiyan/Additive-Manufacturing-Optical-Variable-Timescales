# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:25:55 2024
@author: vigneashwara.p
email: vigneashwara.pandiyan@tii.ae


_status_: "Prototyping"
_maintainer_ = Vigneashwara Pandiyan
-made_with_ = AMRCWS14

Modification and reuse of this code should be authorized by the first owner, code author(s) 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import os


def Euclidean_2D(epoch_length, folder_created, filename):
    """
    Calculate the distance between centroids and data samples for different layer thicknesses.
    Args:
        epoch_length (int): The epoch length.
        folder_created (str): The path to the folder where the data is stored.
        filename (str): The name of the file.
    Returns:
        None
    Raises:
        None
    """
    # This function calculates the distance between centroids and data samples for different layer thicknesses.
    # It takes the epoch length, folder path, and filename as input.
    # The function does not return any value, it plots the results and saves the plot as an image.

    train_embeddings = folder_created+'/' + \
        str(filename)+'_TSNE_'+str(epoch_length) + '.npy'
    features = np.load(train_embeddings)
    train_labelsname = folder_created+'/' + \
        str(filename)+'_label_'+str(epoch_length)+'.npy'
    labels = np.load(train_labelsname)

    # Define the new labels for the classes
    new_labels = ['10 um', '20 um', '30 um', '40 um', '50 um', '60 um',
                  '70 um', '80 um', '90 um', '100 um', '110 um']

    colors = cm.rainbow(np.linspace(0, 1, len(new_labels)))

    # Calculate the centroid for each class
    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        class_points = features[labels == label]
        centroid = np.mean(class_points, axis=0)
        centroids.append(centroid)

    # Calculate the distance from each centroid to other class centroids
    centroid_distances = {}

    for i, label in enumerate(unique_labels):
        distances = []
        for j, other_label in enumerate(unique_labels):
            if i != j:
                distance = euclidean(centroids[i], centroids[j])
                distances.append(distance)
            else:
                distances.append(0)  # distance to itself is 0
        centroid_distances[label] = distances

    # Calculate standard deviations for each class to add as error bars
    centroid_std_devs = {}

    for i, label in enumerate(unique_labels):
        distances = []
        class_points = features[labels == label]
        for j, other_label in enumerate(unique_labels):
            other_points = features[labels == other_label]
            if i != j:
                std_dev = np.std(np.linalg.norm(
                    class_points - np.mean(other_points, axis=0), axis=1))
                distances.append(std_dev)
            else:
                # standard deviation for distance to itself is 0
                distances.append(0)
        centroid_std_devs[label] = distances

    n_classes = len(new_labels)
    n_cols = 2

    if n_classes//2 == 0:
        n_rows = (n_classes) // n_cols
    else:
        n_rows = (n_classes + 1) // n_cols  # Calculate the number of rows

    # Plot all 11 graphs in a 3-row by 5-column layout
    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs = GridSpec(n_rows, n_cols, figure=fig)
    # axes = axes.flatten()  # Flatten the axes array to easily iterate
    axes = []
    for i, label in enumerate(unique_labels):

        if n_classes % n_cols != 0 and i == n_classes - 1:

            ax = plt.subplot2grid((n_rows, n_classes),
                                  (i//n_cols, 2), colspan=6)
            pos = ax.get_position()  # Get the original position
            new_pos = [pos.x0 + 0.05, pos.y0 - 0.17, pos.width,
                       pos.height]  # Shift to the left by 0.05
            ax.set_position(new_pos)

        else:
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols])

        axes.append(ax)
        bars = ax.bar(new_labels, centroid_distances[label],
                      yerr=centroid_std_devs[label], capsize=4,
                      color=colors)

        bars[label].set_color('black')
        ax.set_title(f'Layer thickness: {new_labels[label]}', fontsize=15)
        ax.tick_params(axis='x', labelsize=14)  # Change font size for x-ticks
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xticklabels(new_labels, rotation=45, ha='right')
        ax.set_xlabel('Layer thickness', fontsize=15)
        ax.set_ylabel('Distance (a.u)', fontsize=15)

    graph_title = 'Distance between centroids and data samples for different layer thicknesses' + "\n" + \
        'on window length of '+str(1000-epoch_length)

    plt.suptitle(graph_title, fontsize=15)
    plot = 'Euclidean_' + str(epoch_length) + '_plot.png'
    plt.savefig(os.path.join(folder_created, plot),
                bbox_inches='tight', dpi=800)
    plt.show()



    
