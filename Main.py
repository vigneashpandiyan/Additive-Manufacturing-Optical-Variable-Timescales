# -*- coding: utf-8 -*-
"""
@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch,vigneashpandiyan@gmail.com

The codes in the following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"

@any reuse of this code should be authorised by the code author
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
from Centroid_calculation.Utils_centroid import *

from Inference.Latency import *

from Dataloader.Optical_dataset_loader import *
from Trainer.Selftrain import encoder_backbone, count_parameters
from Visualisation.Temporal_latent_plots import latent_plots, plot_windows
from Visualisation. t-SNE import tsne_visualization
from Evaluation.Backbone_evaluation import Encoder_backbone_evaluation
from Model.Network import TemporalCNN
from Parser. parser import parse_option

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
path = r'C:\Cloud\Github\Additive-Manufacturing-Optical-Variable-Timescales\Dataset'
dataset_name = 'D1_rawspace_1000.npy'
dataset_label = 'D1_classspace_1000.npy'
window_size = 1000
opt = parse_option()
print(opt.class_type)

# %%
# Augmentation type
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
# Folder creation for saving the results, filename = 'PhotodiodeD1'
folder_created = os.path.join('Figures/', filename)
print(folder_created)
# %%
# Loading dataset and splitting the data into train, validation and test sets

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
# Visualisation of the latent space across different window lengths.
window_lengths = [0, 100, 250, 500]

# %%
for length in window_lengths:
    # Plotting the latent space using and computing them on specific window lengths
    X, y, axe, fig, graph_name = tsne_visualization(
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
# %% Finetuning for the downstream task
# Latent vectors are computed, and the NN classifier is trained on top of them with the known ground truth
acc_test, epoch_max_point = Encoder_backbone_evaluation(
    x_train, y_train, x_val, y_val, x_test, y_test, nb_class, ckpt,
    opt, folder_created, filename, 0, None)

# %%
# Counting the number of parameters in the model
backbone = TemporalCNN(opt.feature_size).cuda()
count_parameters(backbone)
# %%
''' Plotting functions'''
# Plotting the latent spaces computed from the backbone individually in 2D and 3D, driven by the window lengths
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
# Compute the centroids of the latent space corresponding to classes that are in the centred, filename=filename, epoch_length=length)

# %%
# Compute the centroids of the latent space corresponding to classes that are in the centre/ and extremes
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

#%%
    
for epoch_length in window_lengths:
    
    results = anomaly_detection_centroid(epoch_length, folder_created,filename)
    
#%% Inference 
filename_ = 'Preprocessed'
folder_created_ = os.path.join('Figures/', filename_)
print(folder_created_)
timing_results = []
for epoch_length in window_lengths:
    result = Inference_calc(x_train, y_train, x_val, y_val, x_test, y_test, ckpt, opt, folder_created_, str(filename), epoch_length)
    train_embeddings, val_embeddings, test_embeddings, train_labelsname, val_labelsname, test_labelsname, timing_stats = result
    
    print(f"Epoch Length {1000-epoch_length}:")
    print(f"  Train Time: {timing_stats['train'][0]:.6f} ± {timing_stats['train'][1]:.6f} sec/sample")
    print(f"  Val   Time: {timing_stats['val'][0]:.6f} ± {timing_stats['val'][1]:.6f} sec/sample")
    print(f"  Test  Time: {timing_stats['test'][0]:.6f} ± {timing_stats['test'][1]:.6f} sec/sample")

    timing_results.append((1000-epoch_length, timing_stats))
    
    epoch_lengths = [item[0] for item in timing_results]
    train_means = np.array([item[1]['train'][0] for item in timing_results])   # convert to ms
    val_means   = np.array([item[1]['val'][0] for item in timing_results]) 
    test_means  = np.array([item[1]['test'][0] for item in timing_results]) 

    train_stds = np.array([item[1]['train'][1] for item in timing_results]) 
    val_stds   = np.array([item[1]['val'][1] for item in timing_results]) 
    test_stds  = np.array([item[1]['test'][1] for item in timing_results]) 

    # Bar width and positions
    bar_width = 0.25
    x = np.arange(len(epoch_lengths))

    # Create the plot
    plt.figure(figsize=(6, 4),dpi=200)
    plt.bar(x - bar_width, train_means, yerr=train_stds, width=bar_width, label='Train set', capsize=4)
    plt.bar(x,              val_means,   yerr=val_stds,   width=bar_width, label='Validation set', capsize=4)
    plt.bar(x + bar_width, test_means,  yerr=test_stds,  width=bar_width, label='Test set', capsize=4)

    # Axis formatting
    # epoch_lengths = 1000-epoch_lengths
    plt.xticks(ticks=x, labels=epoch_lengths)
    plt.xlabel("Window Length")
    plt.ylabel("Processing time per sample (ms)")
    plt.title("Per-sample inference time vs window Length")
    plt.legend()
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_created_, f"distance_histogram_epoch.png"))
    plt.show()
    
