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
import scipy.signal as signal
from Utils_freqfeatures import *
import os
import pandas as pd
#import librosa
print(np.__version__)

# %%
file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)
folder_name = 'Windowed'
sample_rate = 100000
windowsize = 1000
N = windowsize
t0 = 0
dt = 1/sample_rate
time = np.arange(0, N) * dt + t0
band_size = 6
# %%

path_ = os.path.join(total_path, folder_name)
print("Name of the folder..", path_)

try:
    os.makedirs(path_, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")

# %%

path = r'C:\Users\srpv\Desktop\ETH zurich-Dataset 5\LPBF Temporal Self-Time learning\Data'
dataset_name = 'D1_rawspace_1000.npy'
dataset_label = 'D1_classspace_1000.npy'
print("dataset_path...", path)
print("dataset_name...", dataset_name)

rawspace = np.load("{}/{}".format(path, dataset_name))
classspace = np.load("{}/{}".format(path, dataset_label))
rawspace = normalize_to_minus_one(rawspace)


# %%

featurespace = Freqfunction(rawspace, sample_rate, band_size)
featurespace = np.asarray(featurespace)
featurespace = np.squeeze(featurespace)
featurefile = 'Feature_FFT'+'_' + str(windowsize)+'.npy'
np.save(featurefile, featurespace, allow_pickle=True)
