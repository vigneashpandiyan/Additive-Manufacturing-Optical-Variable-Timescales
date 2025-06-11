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
import argparse


def parse_option():

    # Training parameters

    parser = argparse.ArgumentParser('argument for training')

    # Bigger is better.
    parser.add_argument('--K', type=int, default=8,
                        help='Number of augmentation for each sample')

    parser.add_argument('--feature_size', type=int, default=32,
                        help='feature_size')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')

    parser.add_argument('--patience', type=int, default=50,
                        help='training patience')

    parser.add_argument('--aug_type', type=str,
                        default='none', help='Augmentation type')

    parser.add_argument('--piece_size', type=float, default=0.2,
                        help='piece size for time series piece sampling')

    parser.add_argument('--class_type', type=str,
                        default='3C', help='Classification type')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')

    # Testing parameters
    parser.add_argument('--learning_rate_test', type=float, default=0.01,
                        help='learning_rate_test')

    parser.add_argument('--patience_test', type=int, default=100,
                        help='number of training patience')

    # Training and Testing

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    parser.add_argument('--bayesian_train_size', type=int, default=1,
                        help='bayesian_train_size')  # default training

    parser.add_argument('--epochs_test', type=int, default=100,
                        help='number of test epochs')

    # parser.add_argument('--bayesian_size', type=int, default=1,
    #                     help='bayesian_size')

    opt = parser.parse_args()
    return opt
