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
import torch.utils.data as data


class LPBF_loader_eval(data.Dataset):
    """
    This class represents a dataset loader for LPBF evaluation data.

    Args:
        data (array-like): The input data array.
        targets (array-like): The target array.
        transform (callable, optional): A function/transform to be applied on the input data.
        n_epoch (int): The number of epochs.

    Returns:
        tuple: A tuple containing the transformed input data and the target.

    """

    def __init__(self, data, targets, transform, n_epoch):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform
        self.n_epoch = n_epoch

    def __getitem__(self, index):
        ts, target = self.data[index], self.targets[index]

        ts = ts[self.n_epoch:]

        # ts_mean = np.max(ts)
        # ts_std = np.std(ts)
        # ts = (ts - ts_mean) / ts_std

        if self.transform is not None:
            ts_transformed = self.transform(ts.copy())
        else:
            ts_transformed = ts

        return ts_transformed, target

    def __len__(self):
        return self.data.shape[0]
