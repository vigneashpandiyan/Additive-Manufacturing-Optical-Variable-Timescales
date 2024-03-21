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
import torch.utils.data as data


class LPBF_loader_eval(data.Dataset):

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
