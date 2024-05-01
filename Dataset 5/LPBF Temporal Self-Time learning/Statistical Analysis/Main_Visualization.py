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

# %%
# Get the path of the current working directory
file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)
#%%
# Create a folder to save the data
folder_name = 'Visualization_Plots'
path_ = os.path.join(total_path, folder_name)
print("Name of the folder..", path_)

try:
    os.makedirs(path_, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")

#%%
# Load the dataset with frequency domain features
data_feature = os.path.join(total_path, 'Feature_FFT_1000.npy')
print(data_feature)
data_feature = np.load(data_feature)
print(data_feature.shape[1])
data_feature = pd.DataFrame(data_feature)

# Load the dataset with ground truth labels
data_class = os.path.join(total_path, 'classpace_1000.npy')
print(data_class)
data_class = np.load(data_class)
print(data_class.shape)
data_class = pd.DataFrame(data_class)
data_class.columns = ['Categorical']
data_class = pd.DataFrame(data_class)

# Concatenate the feature and class labels
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
# Plotting the features in the EDA analysis
def plots(i, Featurespace, classspace, feature, path_):

    """
    This function plots various visualizations for exploratory data analysis (EDA).
    
    Parameters:
        i (int): Index of the data point to plot.
        Featurespace (pd.DataFrame): DataFrame containing the feature data.
        classspace (pd.DataFrame): DataFrame containing the class labels.
        feature (str): Name of the feature to plot.
        path_ (str): Path to save the plots.
        
    Returns:
        pd.DataFrame: DataFrame containing the plotted data.
    """
    
    data = (Featurespace[i])
    data = data.astype(np.float64)
    df1 = pd.DataFrame(data)
    df1.rename(columns={df1.columns[0]: "Feature"}, inplace=True)
    df2 = pd.DataFrame(classspace)
    df2.rename(columns={df2.columns[0]: "Categorical"}, inplace=True)
    data = pd.concat([df1, df2], axis=1)

    label_custom = ['10 µm', '20 µm', '30 µm',
                    '40 µm', '50 µm', '60 µm', '70 µm', '80 µm', '90 µm', '100 µm', '110 µm']

    kdeplot(data, feature, path_, label_custom)
    hist(data, feature, path_, label_custom)

    df2 = df2['Categorical'].replace(0, '10 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(1, '20 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(2, '30 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(3, '40 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(4, '50 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(5, '60 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(6, '70 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(7, '80 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(8, '90 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(9, '100 µm')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(10, '110 µm')
    df2 = pd.DataFrame(df2)

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
data = plots(3, Featurespace, classspace, "RMS", path_)  # RMS-3 , #Skewness-6,  #Kurtosis-7
data = plots(6, Featurespace, classspace, "Skewness", path_)  # RMS-3 , #Skewness-6,  #Kurtosis-7
data = plots(7, Featurespace, classspace, "Kurtosis", path_)  # RMS-3 , #Skewness-6,  #Kurtosis-7
