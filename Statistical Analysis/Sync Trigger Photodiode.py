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
numbers = re.compile(r'(\d+)')


def numericalSort(value):
    """
    Sorts a list of strings containing numbers in a natural order.

    Args:
        value (str): The string to be sorted.

    Returns:
        list: The sorted list of strings.
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

# %%
# Get the path of the current working directory
file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)
# %%
# Create a folder to save the data
folder_name = 'Visualization_Plots'
path_ = os.path.join(total_path, folder_name)
print("Name of the folder..", path_)

try:
    os.makedirs(path_, exist_ok=True)
    print("Directory created....")
except OSError as error:
    print("Directory already exists....")

# %%
#  Path to the data folder and dataset name
path = r'C:\Users\srpv\Desktop\C4Science\lpbf-optical-variable-time-scales\LPBF Temporal Self-Time learning\Raw_Data'
sample_rate = 100000  # 100 kHz

# %%

r = os.listdir(path_)
print(r)
all_files = glob.glob(path + "/*.pcd")
print("Total files in the folder....", len(all_files))
numbers = re.compile(r'(\d+)')

for filename in sorted(all_files, key=numericalSort):

    print(filename)

    cols = ['t', 'x', 'y', 'z', 'intensity', 'sensor1',
            'sensor2', 'sensor3', 'status', 'controller']
    data = pd.read_csv(filename, delimiter=" ", skiprows=26,
                       dtype=np.int32, names=cols)
    # sig = pd.read_csv(filename, delimiter=" ", dtype=np.int32)

    intensity = data['intensity'].values
    intensity = pd.DataFrame(intensity)

    trigger = data['status'].values
    trigger = pd.DataFrame(trigger)


vertical_stack = pd.concat([trigger, intensity], axis=1)
# chop the dataset
vertical_stack = vertical_stack.iloc[0:80000, 0:2]


def plot_rawdata(path_, sample_rate, vertical_stack):
    vertical_stack = np.array(vertical_stack)
    Controller_trigger, Intensity = vertical_stack[:, 0], vertical_stack[:, 1]

    t0 = 0
    dt = 1/sample_rate
    N = Controller_trigger.shape[0]
    time = np.arange(0, N) * dt + t0

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    ax1.plot(time, Controller_trigger, 'red')
    ax1.set_ylabel('Amplitude (V)', labelpad=11)
    ax1.legend(['Laser trigger'], loc='upper right',
               frameon=False, bbox_to_anchor=(0.42, 1.325))

    ax2.plot(time, Intensity, 'blue')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Amplitude (mV)')
    ax2.legend(['Photodiode signature'], loc='lower right',
               frameon=False, bbox_to_anchor=(1.0, 2.165))

    plotname = "Synchronization_seg.png"
    plt.savefig(os.path.join(path_, plotname), dpi=800, bbox_inches='tight')
    plt.show()


plot_rawdata(path_, sample_rate, vertical_stack)
