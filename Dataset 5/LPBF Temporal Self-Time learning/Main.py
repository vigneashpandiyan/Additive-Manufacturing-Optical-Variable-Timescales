# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch
The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"
@any reuse of this code should be authorized by the code author
"""

import numpy as np
from Centroid_calculation.Utils_tSNE import tsne_visualization_
from Centroid_calculation.Utils_Latent import *
from Dataloader.Optical_dataset_loader import *
from Trainer.Selftrain import encoder_backbone, count_parameters
from Visualization.Temporal_latent_plots import latent_plots, plot_windows
from Visualization.tSNE import tsne_visualization
from Evaluation.Backbone_evaluation import Encoder_backbone_evaluation
from Model.Network import TemporalCNN
import matplotlib.pyplot as plt
from Parser.parser import parse_option
import matplotlib.animation as animation
import os
import pandas as pd
import torch
import seaborn as sns

torch.cuda.is_available()


# %%
Seeds = [0, 1, 2, 3, 4]
for seed in Seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)

# %%
path = r'C:\Users\srpv\Desktop\ETh zurich-optical-variable-window-length\LPBF Temporal Self-Time learning\Data'
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

folder_created = os.path.join('Figures/', 'Clustering_D1')
print(folder_created)

# %%
# Loading dataset

x_train, y_train, x_val, y_val, x_test, y_test, nb_class = load_LPBF(
    path, dataset_name, dataset_label)

# %%
# Model training

model, Times, path,  Training_accuracy, Training_loss, Training_loss_mean, Training_loss_std = encoder_backbone(
    x_train, y_train, opt, 'Clustering_D1', window_size)

# %%

count_parameters(model)
plot_windows(Times, window_size, folder_created)

ckpt = '{}/backbone_best.tar'.format(folder_created)
lkpt = '{}/Clustering_D1_linear.tar'.format(folder_created)

# %%
# Vizualization of the latent space.

window_lengths = [0, 250, 500]
for seed in window_lengths:
    X, y, ax, fig, graph_name = tsne_visualization(
        x_train, y_train, x_val, y_val, x_test, y_test, ckpt, opt, filename='Clustering_D1', epoch_length=seed)

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))

# %%
latent_plots('Clustering_D1', 0)

# %% Finetunning for the down streaming task

acc_test, epoch_max_point = Encoder_backbone_evaluation(
    x_train, y_train, x_val, y_val, x_test, y_test, nb_class, ckpt,
    opt, 'Clustering_D1', 0, None)

# %%
backbone = TemporalCNN(opt.feature_size).cuda()
count_parameters(backbone)
# %%
''' Plotting functions'''

window_lengths = [0, 250, 500]

# folder_created = r'C:\Users\vigneashwara.p\OneDrive - Technology Innovation Institute\Desktop\Centroid calculation\Data'
filename = 'Clustering_D1'

for epoch_length in window_lengths:

    train_embeddings = folder_created+'/' + \
        str(filename)+'_'+str(epoch_length)+'_embeddings' + '.npy'
    X = np.load(train_embeddings)
    train_labelsname = folder_created+'/' + \
        str(filename)+'_'+str(epoch_length)+'_labels'+'.npy'
    Y = np.load(train_labelsname)

    _, _, ax, fig, graph_name = tsne_visualization_(
        X, Y, filename=filename, epoch_length=epoch_length)

# %%

for epoch_length in window_lengths:

    features, class_labels = plot_latent_2D(epoch_length, folder_created, filename)

    train_embeddings = folder_created+'/' + \
        str(filename)+'_TSNE_'+str(epoch_length) + '.npy'
    features = np.load(train_embeddings)
    train_labelsname = folder_created+'/' + \
        str(filename)+'_label_'+str(epoch_length)+'.npy'

    class_labels = np.load(train_labelsname)
    classes = np.unique(class_labels)

    group = [0, 5, 10]
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
