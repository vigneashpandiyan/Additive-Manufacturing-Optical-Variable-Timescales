# -*- coding: utf-8 -*-
"""
@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch,vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"

@any reuse of this code should be authorized by the code author
"""
#%%
#Libraries to import

import numpy as np
import torch.utils.data as data


class LPBF_Inter_Intra_reasoning(data.Dataset):
    """
    This class represents a dataset for LPBF Inter-Intra reasoning.

    Args:
        data (array-like): The input data.
        targets (array-like): The target values.
        K (int): The total number of augmentations.
        transform (callable): A function/transform to apply to the data.
        transform_cut (callable): A function/transform to apply to the transformed data.
        totensor_transform (callable): A function/transform to convert the data to tensors.
        n_epoch (int): The number of epochs.
        starting_p (int): The starting point.
        stopping_p (int): The stopping point.
        window_size (int): The window size.

    Returns:
        tuple: A tuple containing the transformed data, cut data, labels, and target value.
    """

    def __init__(self, data, targets, K, transform, transform_cut, totensor_transform, n_epoch, starting_p, stopping_p, window_size):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int16)
        self.K = K  # tot number of augmentations
        self.transform = transform
        self.transform_cut = transform_cut
        self.totensor_transform = totensor_transform
        self.n_epoch = n_epoch
        self.starting_p = starting_p
        self.stopping_p = stopping_p
        self.window_size = window_size

    def __getitem__(self, index):

        ts, target = self.data[index], self.targets[index]
        stopping_length = self.starting_p + (self.window_size-self.n_epoch)  # Length of anchors
        ts = ts[self.starting_p:stopping_length+1]

        ts_list = list()
        ts_list0 = list()
        ts_list1 = list()
        label_list = list()

        if self.transform is not None:
            for _ in range(self.K):

                # write a block to truncate the data

                ts_transformed = self.transform(ts.copy())

                # print(ts_transformed.shape) #check the shape of data
                ts_cut0, ts_cut1, label = self.transform_cut(ts_transformed)
                ts_list.append(self.totensor_transform(ts_transformed))
                ts_list0.append(self.totensor_transform(ts_cut0))
                ts_list1.append(self.totensor_transform(ts_cut1))
                label_list.append(label)

        return ts_list, ts_list0, ts_list1, label_list, target

    def __len__(self):
        return self.data.shape[0]
