# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch
The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"
@any reuse of this code should be authorized by the code author
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import glob
import struct
import numpy as np
import functools
import glob
import seaborn as sns

import matplotlib.pyplot as plt
import re
import ntpath
from pathlib import Path
import os
from scipy import signal
from itertools import groupby
from operator import itemgetter
import pywt
import re
# laserpower_scanspeed.pkl per each trial.


# %%
windowsize = 1000

# %%


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def addlabels(k):
    print(k)
    return [k]


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


# def normalize(ts, Features_train_max, Features_train_min):
#     ts = (ts - np.mean(ts)) / (Features_train_max - Features_train_min)
#     return ts

# def normalize(ts,Features_train_max,Features_train_min):


#     ts = 2. * (ts - Features_train_min) / (Features_train_max - Features_train_min) - 1.

#     return ts

def data_clipping(data, windowsize):

    data = data.to_numpy()
    channel = []
    # channel = np.zeros((window_size,0))

    if data.shape[0] >= windowsize:

        window = round(data.shape[0]/windowsize)
        signal_chop = np.split(data, window)
        window = len(signal_chop)
        # print("Total windows: ",data.shape[0])
        print("Total windows: ", window)

        for i in range(window):

            signal_window = signal_chop[i]
            signal_window = np.asarray(signal_window)
            signal_window = np.squeeze(signal_window)
            # print("Length of the array2",signal_window.shape)
            # signal_window = np.asarray(signal_window)
            # print("Length of the array2",signal_window.shape)
            # signal_window=np.expand_dims(signal_window, axis=1)
            channel.append(signal_window)

        # signal_window=signal_chop[0]
        # signal_window = np.asarray(signal_window)
        # signal_window = np.squeeze(signal_window)
        # channel.append(signal_window)

    channel = np.asarray(channel)

    return channel

# %%


rawdataset = np.empty([0, windowsize])
classes = np.empty([0, 1])
j = 0

path_ = r'C:\Users\srpv\Desktop\ETh zurich-optical-variable-window-length\Dataset\Data'
r = os.listdir(path_)

for folder_name in r:
    path = os.path.join(path_, folder_name)

    print(path)

    isDirectory = os.path.isdir(path)
    # path = r'C:\Users\srpv\Desktop\ETHZurich\Dataset 3\Dataset\Data\90_microns' # use your path!

    all_files = glob.glob(path + "/*.pcd")
    numbers = re.compile(r'(\d+)')

    for filename in sorted(all_files, key=numericalSort):

        print(filename)

        cols = ['t', 'x', 'y', 'z', 'intensity', 'sensor1',
                'sensor2', 'sensor3', 'status', 'controller']
        sig = pd.read_csv(filename, delimiter=" ", skiprows=26,
                          dtype=np.int32, names=cols)

        # sig = pd.read_csv(filename, delimiter=" ", dtype=np.int32)

        Trigger = sig.iloc[:, [8]]
        reqd_Index = Trigger[Trigger["status"] >= 0.2].index.tolist()  # Threshold
        reqd_Index = np.array(reqd_Index)

        laser_on = consecutive(reqd_Index)
        del reqd_Index
        rows_computed = 0

        Pyro = sig.iloc[:, [4]]

        for i in laser_on:

            e = len(i)
            e = round(0.01*e)  # Giving enough tolerance for stable signal
            a = np.amin(i)
            a = a+e  # starting point with tolerance (lower limit)
            b = np.amax(i)
            b = b-e  # stopping point with tolerance
            c = b-a

            if c >= windowsize:

                print("into the loop")

                rows, d = divmod(c, windowsize)  # addwindow #check reminder
                a = a+d  # pass the reminder to the starting (lower limit)
                data_stream_trigger = Pyro.iloc[a:b]
                arr = data_clipping(data_stream_trigger, windowsize)
                rawdataset = np.concatenate([rawdataset, arr], axis=0)

                k = addlabels(j)
                classes = np.append(classes, k)

    j = j+1


# windowsize = 5000
# Features_train_max = np.max(rawdataset)
# Features_train_min = np.min(rawdataset)
# rawdataset=normalize(rawdataset,Features_train_max,Features_train_min)
classfile = 'D1_classspace'+'_' + str(windowsize)+'.npy'
classspace = classes.astype(np.float64)
rawfile = 'D1_rawspace'+'_' + str(windowsize)+'.npy'
rawspace = rawdataset.astype(np.float64)
np.save(classfile, classspace, allow_pickle=True)
np.save(rawfile, rawspace, allow_pickle=True)
