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
import os
import seaborn as sns
import joypy
from matplotlib import cm
from matplotlib import colors
from Utils_visualization import *

# %%

sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)
folder_name = 'Visualization_Plots'
path_ = os.path.join(total_path, folder_name)
print("Name of the folder..", path_)

try:
    os.makedirs(path_, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")


data_feature = os.path.join(total_path, 'Feature_FFT_1500.npy')
print(data_feature)
data_feature = np.load(data_feature)
print(data_feature.shape[1])
data_feature = pd.DataFrame(data_feature)
data_class = os.path.join(total_path, 'classpace_1500.npy')
print(data_class)
data_class = np.load(data_class)
print(data_class.shape)
data_class = pd.DataFrame(data_class)
data_class.columns = ['Categorical']

data_class = pd.DataFrame(data_class)


# %%


data = pd.concat([data_feature, data_class], axis=1)
print("Respective windows per category", data.Categorical.value_counts())
minval = min(data.Categorical.value_counts())
# minval=np.round(minval,decimals=-3)
print("windows of the class: ", minval)
data_1 = pd.concat([data[data.Categorical == cat].head(minval)
                   for cat in data.Categorical.unique()])
print("Balanced dataset: ", data_1.Categorical.value_counts())

Featurespace = data_1.iloc[:, :-1]
classspace = data_1.iloc[:, -1]


# %%

def plots(i, Featurespace, classspace, feature, path_):
    # Featurespace = Featurespace.transpose()
    data = (Featurespace[i])
    data = data.astype(np.float64)
    #data= abs(data)
    df1 = pd.DataFrame(data)
    df1.rename(columns={df1.columns[0]: "Feature"}, inplace=True)
    df2 = pd.DataFrame(classspace)

    df2.rename(columns={df2.columns[0]: "Categorical"}, inplace=True)
    data = pd.concat([df1, df2], axis=1)

    label_custom = ['Remelt', '10 µm', '20 µm', '30 µm',
                    '40 µm', '50 µm', '60 µm', '70 µm', '80 µm', '90 µm']

    kdeplot(data, feature, path_, label_custom)
    hist(data, feature, path_, label_custom)

    df2 = df2['Categorical'].replace(0, 'Remelt')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(1, '10 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(2, '20 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(3, '30 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(4, '40 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(5, '50 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(6, '60 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(7, '70 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(8, '80 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(9, '90 µm')
    data = pd.concat([df1, df2], axis=1)

    violinplot(data, feature, path_)
    boxplot(data, feature, path_)
    Histplotsplit(data, feature, path_)
    barplot(data, feature, path_)
    ridgeplot(data, feature, path_)
    kdeplotsplit(data, feature, path_)

    return data


# %%
    # 4#7
data = plots(7, Featurespace, classspace, "Kurtosis", path_)  # RMS-3 , #Skewness-6,  #Kurtosis-7
