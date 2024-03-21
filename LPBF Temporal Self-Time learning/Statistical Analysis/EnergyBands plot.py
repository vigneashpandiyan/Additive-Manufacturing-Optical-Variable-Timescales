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
import seaborn as sns
from scipy.stats import norm
# import joypy
from matplotlib import cm
from scipy import signal
import matplotlib.patches as mpatches
from matplotlib import colors
import os

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
data_class = data_class['Categorical'].replace(0, 'Remelt')
data_class = pd.DataFrame(data_class)
data_class = data_class['Categorical'].replace(1, '10 µm')
data_class = pd.DataFrame(data_class)
data_class = data_class['Categorical'].replace(2, '20 µm')
data_class = pd.DataFrame(data_class)
data_class = data_class['Categorical'].replace(3, '30 µm')
data_class = pd.DataFrame(data_class)
data_class = data_class['Categorical'].replace(4, '40 µm')
data_class = pd.DataFrame(data_class)
data_class = data_class['Categorical'].replace(5, '50 µm')
data_class = pd.DataFrame(data_class)
data_class = data_class['Categorical'].replace(6, '60 µm')
data_class = pd.DataFrame(data_class)
data_class = data_class['Categorical'].replace(7, '70 µm')
data_class = pd.DataFrame(data_class)
data_class = data_class['Categorical'].replace(8, '80 µm')
data_class = pd.DataFrame(data_class)
data_class = data_class['Categorical'].replace(9, '90 µm')
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

classspace = classspace.to_numpy()
Featurespace = Featurespace.to_numpy()

feature = ['0-10 kHZ', '10-20 kHZ', '20-30 kHZ', '30-40 kHZ', '40-50 kHZ']

len(feature)

Featurespace = Featurespace[:, 0:len(feature)]
# %%


def boxcomparisonplots(y_pred, y_true, path):

    Featurespace = pd.DataFrame(y_pred)
    classspace = pd.DataFrame(y_true)
    classspace.columns = ['Categorical']

    uniq = np.sort(classspace.Categorical)
    classes = np.unique(classspace.Categorical)
    color = iter(cm.rainbow(np.linspace(0, 1, len(classes))))

    data = pd.concat([Featurespace, classspace], axis=1)
    minval = min(data.Categorical.value_counts())
    print(minval)
    data = pd.concat([data[data.Categorical == cat].head(minval)
                     for cat in data.Categorical.unique()])

    Featurespace = data.iloc[:, :-1]
    classspace = data.iloc[:, -1]
    values, counts = np.unique(classspace, return_counts=True)
    print(values, counts)

    classspace = classspace.to_numpy()
    Featurespace = Featurespace.to_numpy()

    c = len(Featurespace)
    df1 = pd.DataFrame(Featurespace)
    df1 = np.ravel(df1, order='F')
    df1 = pd.DataFrame(df1)

    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']

    filename = '0-10 kHZ'
    numbers = np.random.randn(c)
    df3 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df3 = df3.drop(['numbers'], axis=1)

    filename = '10-20 kHZ'
    numbers = np.random.randn(c)
    df4 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df4 = df4.drop(['numbers'], axis=1)

    filename = '20-30 kHZ'
    numbers = np.random.randn(c)
    df5 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df5 = df5.drop(['numbers'], axis=1)

    filename = '30-40 kHZ'
    numbers = np.random.randn(c)
    df6 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df6 = df6.drop(['numbers'], axis=1)

    filename = '40-50 kHZ'
    numbers = np.random.randn(c)
    df7 = pd.DataFrame({'labels': filename, 'numbers': numbers})
    df7 = df7.drop(['numbers'], axis=1)

    Energyband = np.concatenate((df3, df4, df5, df6, df7), axis=0)
    Modes = np.concatenate((df2, df2, df2, df2, df2), axis=0)

    Energybands = np.concatenate((df1, Energyband, Modes), axis=1)
    Energybands = pd.DataFrame(Energybands)
    Energybands.columns = ['Frequency distribution', 'Frequency levels', 'Categorical']

    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid", {'axes.grid': False})
    ax = sns.catplot(y="Frequency levels", x="Frequency distribution", hue="Categorical", kind="bar", data=Energybands, height=12,
                     aspect=1.5, palette=color)

    ax._legend.remove()
    plt.legend(loc='lower right', frameon=False, fontsize=40)
    plt.title('Powerspectral density distribution', fontsize=50, pad=10,)
    plt.xlabel('Relative density', labelpad=10, fontsize=40)
    plt.xticks(fontsize=30)

    plt.ylabel('Frequency levels', labelpad=10, fontsize=40)
    plt.yticks(fontsize=40)

    plt.ticklabel_format(style='plain', axis='x')

    plotname = "Frequency distribution"+'.png'
    plt.savefig(os.path.join(path_, plotname), dpi=800, bbox_inches='tight')
    plt.show()


y_pred = Featurespace
y_true = classspace
boxcomparisonplots(y_pred, y_true, total_path)
