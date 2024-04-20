# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch
The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"
@any reuse of this code should be authorized by the code author
"""
import torch

from Model.Network import TemporalCNN
from Trainer.Relational_Reasoning_Trainer import InterIntra_Train
from Dataloader.LPBF_InterIntra import LPBF_Inter_Intra_reasoning
import Dataloader.transforms as transforms
from Visualization.Training_plots import training_plots
import torch.utils.data as data
from prettytable import PrettyTable


import os


def encoder_backbone(x_train, y_train, opt, graph_name, window_size):

    folder_created = os.path.join('Figures/', graph_name)
    print(folder_created)
    try:
        os.makedirs(folder_created, exist_ok=True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")

    K = opt.K  # 'Number of augmentation for each sample'
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size

    prob = 0.2  # Transform Probability
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': jitter,
                       'cutout': cutout,
                       'scaling': scaling,
                       'magnitude_warp': magnitude_warp,
                       'time_warp': time_warp,
                       'window_slice': window_slice,
                       'window_warp': window_warp,
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = [transforms_list[name] for name in opt.aug_type]
    train_transform = transforms.Compose(transforms_targets)
    tensor_transform = transforms.ToTensor()

    if '3C' in opt.class_type:
        cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
        nb_class = 3
    else:
        cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
        nb_class = 2

    backbone = TemporalCNN(opt.feature_size).cuda()

    model = InterIntra_Train(backbone, feature_size, nb_class).cuda()
    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(folder_created))

    Training_accuracy, Training_loss, Training_loss_mean, Training_loss_std, learning_rate, Times = model.train(x_train, y_train, K,
                                                                                                                train_transform, cut_piece, tensor_transform, batch_size,
                                                                                                                graph_name, tot_epochs=tot_epochs, opt=opt, window_size=window_size)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(folder_created))

    training_plots(Training_loss, Training_accuracy, Training_loss_mean,
                   Training_loss_std, learning_rate, Times, window_size)

    return model, Times, folder_created, Training_accuracy, Training_loss, Training_loss_mean, Training_loss_std


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
