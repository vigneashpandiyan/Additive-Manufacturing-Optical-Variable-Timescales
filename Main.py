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
import matplotlib.animation as animation
import os
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from Centroid_calculation.Utils_tSNE import tsne_visualization_
from Centroid_calculation.Utils_Latent import *
from Centroid_calculation.Utils_plotting import *
from Dataloader.Optical_dataset_loader import *
from Trainer.Selftrain import encoder_backbone, count_parameters
from Visualization.Temporal_latent_plots import latent_plots, plot_windows
from Visualization.tSNE import tsne_visualization
from Evaluation.Backbone_evaluation import Encoder_backbone_evaluation
from Model.Network import TemporalCNN
from Parser.parser import parse_option

# %%
# Checking the GPU availability
print(torch.cuda.is_available())
print(torch.__version__)
Seeds = [0, 1, 2, 3, 4]

for seed in Seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)

# %%
# Path to the data folder and dataset name
# http://dx.doi.org/10.5281/zenodo.11101714
path = r'C:\Users\vigneashwara.p\Desktop\C4Science\lpbf-optical-variable-window-length\Dataset'
dataset_name = 'D1_rawspace_1000.npy'
dataset_label = 'D1_classspace_1000.npy'
window_size = 1000
opt = parse_option()
print(opt.class_type)

# %%
# Augumentation type
aug1 = ['cutout']
aug2 = ['jitter']
aug3 = ['scaling']
aug4 = ['magnitude_warp']
aug5 = ['time_warp']
aug6 = ['window_slice']
aug7 = ['window_warp']
opt.aug_type = aug1 + aug2 + aug3 + aug4 + aug5 + aug6 + aug7
print(opt.aug_type)

# %%
# Folder creation for saving the results
filename = 'PhotodiodeD1'
folder_created = os.path.join('Figures/', filename)
print(folder_created)
# %%
# Loading dataset and splitting the data into train, validation and test set

x_train, y_train, x_val, y_val, x_test, y_test, nb_class = load_LPBF(
    path, dataset_name, dataset_label)

# %%
# Model training and plotting the training curves
model, Times, path,  Training_accuracy, Training_loss, Training_loss_mean, Training_loss_std = encoder_backbone(
    x_train, y_train, opt, str(filename), window_size)

# %%
# Counting the number of parameters in the model and plotting the windows
count_parameters(model)
plot_windows(Times, window_size, folder_created)
# %%
# Saving the model checkpoint
ckpt = '{}/backbone_best.tar'.format(folder_created)
lkpt = '{}/Clustering_D1_linear.tar'.format(folder_created)

# %%
# Vizualization of the latent space across different window lengths.
window_lengths = [0, 100, 250, 500]

# %%
for length in window_lengths:
    # Plotting the latent space using and computing them on specific window lengths
    X, y, ax, fig, graph_name = tsne_visualization(
        x_train, y_train, x_val, y_val, x_test, y_test, ckpt, opt, folder_created=folder_created, filename=filename, epoch_length=length)

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(
        fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))

# %%
# Plotting the latent spaces computed from the backbone individually in 2D and 3D
latent_plots(folder_created, filename, 0)
# %% Finetunning for the down streaming task
# Latent vectors are computed and NN classifier is trained on top of it with the known ground_truth
acc_test, epoch_max_point = Encoder_backbone_evaluation(
    x_train, y_train, x_val, y_val, x_test, y_test, nb_class, ckpt,
    opt, folder_created, filename, 0, None)

# %%
# Counting the number of parameters in the model
backbone = TemporalCNN(opt.feature_size).cuda()
count_parameters(backbone)
# %%
''' Plotting functions'''
# Plotting the latent spaces computed from the backbone individually in 2D and 3D on driven by the window lengths
for length in window_lengths:

    train_embeddings = folder_created+'/' + \
        str(filename)+'_'+str(length)+'_embeddings' + '.npy'
    X = np.load(train_embeddings)
    train_labelsname = folder_created+'/' + \
        str(filename)+'_'+str(length)+'_labels'+'.npy'
    Y = np.load(train_labelsname)

    _, _, ax, fig, graph_name = tsne_visualization_(
        X, Y, folder_created, filename=filename, epoch_length=length)

# %%
# Compute the centroids of the latent space corresponding to classes that in the center/ and extremes
for epoch_length in window_lengths:
    group = [0, 5, 10]
    features, class_labels = plot_latent_2D(
        epoch_length, folder_created, filename, group, window_size)

    train_embeddings = folder_created+'/' + \
        str(filename)+'_TSNE_'+str(epoch_length) + '.npy'
    features = np.load(train_embeddings)
    train_labelsname = folder_created+'/' + \
        str(filename)+'_label_'+str(epoch_length)+'.npy'

    class_labels = np.load(train_labelsname)
    classes = np.unique(class_labels)

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
    group = [1, 2, 3, 4]
    value = 2
    binary_cluster(group, value, data, centroids, epoch_length, window_size)

    group = [6, 7, 8, 9]
    value = 0
    binary_cluster(group, value, data, centroids, epoch_length, window_size)

# %%

for epoch_length in window_lengths:
    Euclidean_2D(epoch_length, folder_created, filename)
