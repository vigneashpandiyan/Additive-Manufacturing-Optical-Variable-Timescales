# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch
The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"
@any reuse of this code should be authorized by the code author
"""

# %
import numpy as np
from matplotlib.pyplot import cm
import itertools
import matplotlib.pyplot as plt
import pandas as pd


def plot_latent_2D(epoch_length, folder_created, filename):
    'Call from main'

    # for epoch_length in window_lengths:

    train_embeddings = folder_created+'/' + \
        str(filename)+'_TSNE_'+str(epoch_length) + '.npy'
    features = np.load(train_embeddings)
    train_labelsname = folder_created+'/' + \
        str(filename)+'_label_'+str(epoch_length)+'.npy'

    class_labels = np.load(train_labelsname)
    classes = np.unique(class_labels)
    group = [0, 4, 9]

    centroids = plot_centroids(features, class_labels, group)
    Featurespace = pd.DataFrame(features)
    classspace = pd.DataFrame(class_labels)
    classspace.columns = ['Categorical']
    data = pd.concat([Featurespace, classspace], axis=1)
    minval = min(data.Categorical.value_counts())
    print("windows of the class: ", minval)
    # minval = 100
    data = pd.concat([data[data.Categorical == cat].head(minval)
                     for cat in data.Categorical.unique()])

    color = iter(cm.rainbow(np.linspace(0, 1, len(group))))
    marker = itertools.cycle(('X', '*', 'o'))
    labels = ['Remelt', '10 um', '20 um', '30 um', '40 um',
              '50 um', '60 um', '70 um', '80 um', '90 um']
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(6, 6), dpi=200)
    j = 0
    for i in group:

        c = next(color)
        m = next(marker)

        indx = (data['Categorical'] == i)
        print(indx)

        x1 = data.iloc[:, 0]
        x2 = data.iloc[:, 1]
        # x2 = x2.to_numpy()
        a = x1[indx]
        b = x2[indx]

        plt.plot(a, b, color=c, label=labels[i], marker=m, linestyle='', ms=10)

    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='D', color='black', label='Centroids', s=100)

    plt.xlabel('Dimension 1', labelpad=15, fontsize=20)
    plt.ylabel('Dimension 2', labelpad=15, fontsize=20)
    title = 'Centroids on lower dimensional embeddings computed' + "\n" + \
        'on window length of '+str(1500-epoch_length)+' data points'
    plt.legend()
    plt.locator_params(nbins=6)

    plt.title(title, fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(bbox_to_anchor=(1, 1), frameon=False, markerscale=1.5, fontsize=20)
    graphtitle = folder_created+'/' + str(epoch_length)+'_Centroid_on data.png'
    plt.savefig(graphtitle, bbox_inches='tight', dpi=200)
    plt.show()

    return features, class_labels


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# Function to find the closest centroid for a given data point


def find_closest_centroid(data_point, centroids):
    distances = [euclidean_distance(data_point, centroid)
                 for centroid in centroids]

    # print(distances)
    closest_centroid_index = np.argmin(distances)
    indices_sorted_by_min_value = sorted(
        range(len(distances)), key=lambda i: distances[i])

    # print(indices_sorted_by_min_value)
    return closest_centroid_index, indices_sorted_by_min_value, distances


def plot_centroids(features, class_labels, group):

    centroids = np.zeros((len(features[0]), len(group)))
    j = 0
    # Compute centroid for each class
    for i in group:  # Assuming there are 4 classes
        # Find indices of rows belonging to class i
        class_indices = np.where(class_labels == i)[0]
        # Extract feature vectors for class i
        class_features = features[class_indices]
        # Compute mean of feature vectors
        class_centroid = np.mean(class_features, axis=0)
        # Assign centroid to centroids array
        centroids[j] = class_centroid
        j = j+1

    # print("Centroids for each class:")
    # print(centroids)

    color = iter(cm.rainbow(np.linspace(0, 1, len(group))))
    marker = itertools.cycle(('X', '*', 'o'))
    labels = ['Remelt', '10 um', '20 um', '30 um', '40 um',
              '50 um', '60 um', '70 um', '80 um', '90 um']

    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(6, 6), dpi=200)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 3
    plt.rc("font", size=23)

    i = 0
    for j in group:

        c = next(color)
        m = next(marker)
        a = centroids[0][i]
        b = centroids[i][1]
        plt.plot(a, b, color=c, label=labels[j],
                 marker=m, linestyle='', ms=25)
        i = i+1

    plt.xlabel('Dimension 1', labelpad=15, fontsize=25)
    plt.ylabel('Dimension 2', labelpad=15, fontsize=25)

    plt.title('Centroids computation', pad=15, fontsize=25)
    # plt.legend()
    plt.locator_params(nbins=6)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(bbox_to_anchor=(1, 1), frameon=False, markerscale=1.2)
    plt.savefig('Centroid.png', bbox_inches='tight', dpi=200)
    plt.show()

    return centroids


def increment_counter(lst, value, my_counter):

    if lst[-1] == value:
        my_counter += 1
    else:
        my_counter += 0

    return my_counter


def binary_cluster(group, value, data, centroids, epoch_length, window_size):

    my_counter = 0
    # global my_counter

    total = 0
    for i in group:

        data_1 = data[data.Categorical == i]

        feature = data_1.iloc[:, :-1]
        label = data_1.iloc[:, -1]

        feature = feature.to_numpy()
        for x in feature:
            total = total+1
            # print(x)
            closest_centroid_index, lst, distances = find_closest_centroid(
                x, centroids)

            my_counter = increment_counter(lst, value, my_counter)

    print("Results on {} with window length {}:" .format(group, window_size-epoch_length))
    print("total entries in the dataset:", total)
    print("correct prediction:", my_counter)
