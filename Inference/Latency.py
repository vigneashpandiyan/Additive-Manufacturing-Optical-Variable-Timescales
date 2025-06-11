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

import os
import time
import numpy as np
import time
import torch

import Dataloader.transforms as transforms
from Dataloader.LPBF_loader_eval import LPBF_loader_eval
import torch.utils.data as data
from Trainer.pytorchtools import EarlyStopping
from Model.Network import TemporalCNN
from Visualization.tSNE_plots import *
from matplotlib import animation


# %%

def process_loader(loader, backbone_lineval, folder_created, filename, epoch_length, name):

    X, y = [], []
    batch_times = []
    for data, target in loader:
        start = time.time()
        data = data.cuda()
        output = backbone_lineval(data)
        end = time.time()

        elapsed = end - start
        batch_size = data.size(0)
        batch_times.append(elapsed / batch_size)

        X.append(output.detach().cpu().numpy())
        y.append(target.detach().cpu().numpy())

    emb = [item for sublist in X for item in sublist]
    lab = [item for sublist in y for item in sublist]

    np.save(f"{folder_created}/{filename}_{epoch_length}_{name}_embeddings.npy", emb)
    np.save(f"{folder_created}/{filename}_{epoch_length}_{name}_labels.npy", lab)

    mean_time = np.mean(batch_times)
    std_time = np.std(batch_times)
    print(f"{name.capitalize()} processing time per sample: {
          mean_time:.6f} ± {std_time:.6f} sec/sample")

    return emb, lab, mean_time, std_time


def Inference_calc(x_train, y_train, x_val, y_val, x_test, y_test, ckpt, opt, folder_created, filename, epoch_length):

    backbone_lineval = TemporalCNN(opt.feature_size).cuda()
    checkpoint = torch.load(ckpt, map_location='cpu')
    backbone_lineval.load_state_dict(checkpoint)
    print('Linear evaluation')
    backbone_lineval.eval()

    transform_lineval = transforms.Compose([transforms.ToTensor()])
    train_set = LPBF_loader_eval(
        x_train, y_train, transform_lineval, n_epoch=epoch_length)
    val_set = LPBF_loader_eval(
        x_val, y_val, transform_lineval, n_epoch=epoch_length)
    test_set = LPBF_loader_eval(
        x_test, y_test, transform_lineval, n_epoch=epoch_length)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False)

    val_emb, val_lab, val_mean, val_std = process_loader(
        val_loader, backbone_lineval, folder_created, filename, epoch_length, "val")
    test_emb, test_lab, test_mean, test_std = process_loader(
        test_loader, backbone_lineval, folder_created, filename, epoch_length, "test")
    train_emb, train_lab, train_mean, train_std = process_loader(
        train_loader, backbone_lineval, folder_created, filename, epoch_length, "train")

    timing_stats = {
        "val": (val_mean, val_std),
        "test": (test_mean, test_std),
        "train": (train_mean, train_std)
    }

    return train_emb, val_emb, test_emb, train_lab, val_lab, test_lab, timing_stats
