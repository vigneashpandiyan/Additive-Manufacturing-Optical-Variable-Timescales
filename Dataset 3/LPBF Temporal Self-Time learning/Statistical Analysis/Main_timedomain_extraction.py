# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch
The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"
@any reuse of this code should be authorized by the code author
"""
# %%
# Libraries to import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils_timefeatures import *
import os
#import librosa
print(np.__version__)

# %% Folder creation

file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)
folder_name = 'Windowed'
sample_rate = 100000
windowsize = 1500
N = windowsize
t0 = 0
dt = 1/sample_rate
time = np.arange(0, N) * dt + t0
# %%


path_ = os.path.join(total_path, folder_name)
print("Name of the folder..", path_)

try:
    os.makedirs(path_, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")
# %%

path = r'C:\Users\srpv\Desktop\C4Science\lpbf-optical-variable-time-scales\LPBF Temporal Self-Time learning\Data'
dataset_name = 'D1_rawspace_1500.npy'
dataset_label = 'D1_classspace_1500.npy'
print("dataset_path...", path)
print("dataset_name...", dataset_name)

rawspace = np.load("{}/{}".format(path, dataset_name))
classspace = np.load("{}/{}".format(path, dataset_label))
rawspace = normalize_to_minus_one(rawspace)


class_file = 'classpace'+'_' + str(windowsize)+'.npy'
np.save(class_file, classspace, allow_pickle=True)

rawspace_ = np.asarray(rawspace)
rawspace_ = np.squeeze(rawspace_)
raw_file = 'Normalised_rawspace'+'_' + str(windowsize)+'.npy'
np.save(raw_file, rawspace_, allow_pickle=True)

# %%

featurespace = Timefunction(rawspace, windowsize)

featurespace = np.asarray(featurespace)
featurespace = np.squeeze(featurespace)
featurefile = 'Featurespace'+'_' + str(windowsize)+'.npy'
np.save(featurefile, featurespace, allow_pickle=True)
