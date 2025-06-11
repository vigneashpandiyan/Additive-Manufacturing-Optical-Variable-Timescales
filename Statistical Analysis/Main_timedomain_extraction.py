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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils_timefeatures import *
import os
print(np.__version__)

# %%
# Get the path of the current working directory
file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)
# %%
# More info on the dataset/sampling rate/ window size
folder_name = 'Windowed'
sample_rate = 100000
windowsize = 1000
N = windowsize
t0 = 0
dt = 1/sample_rate
time = np.arange(0, N) * dt + t0
# %%
# Create a folder to save the data
path_ = os.path.join(total_path, folder_name)
print("Name of the folder..", path_)

try:
    os.makedirs(path_, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")
# %%
# load the dataset and normalize the dataset
# http://dx.doi.org/10.5281/zenodo.11101714
path = r'C:\Cloud\Github\Additive-Manufacturing-Optical-Variable-Timescales\Dataset'
dataset_name = 'D1_rawspace_1000.npy'
dataset_label = 'D1_classspace_1000.npy'
print("dataset_path...", path)
print("dataset_name...", dataset_name)
rawspace = np.load("{}/{}".format(path, dataset_name))
classspace = np.load("{}/{}".format(path, dataset_label))

# Normalize the dataset
rawspace = normalize_to_minus_one(rawspace)
rawspace_ = np.asarray(rawspace)
rawspace_ = np.squeeze(rawspace_)
raw_file = 'Normalised_rawspace'+'_' + str(windowsize)+'.npy'
np.save(raw_file, rawspace_, allow_pickle=True)

class_file = 'classpace'+'_' + str(windowsize)+'.npy'
np.save(class_file, classspace, allow_pickle=True)
# Extract features in time domain
featurespace = Timefunction(rawspace, windowsize)
featurespace = np.asarray(featurespace)
featurespace = np.squeeze(featurespace)
# Save the feature space
featurefile = 'Featurespace'+'_' + str(windowsize)+'.npy'
np.save(featurefile, featurespace, allow_pickle=True)
