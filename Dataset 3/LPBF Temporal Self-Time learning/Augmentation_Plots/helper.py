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
import matplotlib.pyplot as plt
import random


def plot_d(x, t, x_, colour, save_file='', Transformation=''):

    fig = plt.figure(figsize=(9, 4))
    plt.title('Original signal -- ' + str(Transformation), fontsize=15)

    plt.plot(t, x.ravel(), 'black', color='0.7', linewidth=2)
    plt.plot(t, x_.ravel(), colour, ls='--', linewidth=1)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    plt.xlabel("time (ms)", fontsize=15)
    plt.ylabel('Normalized amplitude', fontsize=15)

    plt.legend(['Original signal', str(Transformation)], fontsize=12, loc='lower left')

    # plt.ylim([-0.2, 0.3])
    # plt.ylim([np.min(x)*1.2, np.max(x)*1.2])

    plt.savefig(save_file, dpi=200)
    plt.show()


def plot1d(x, t, save_file=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))

    plt.title('Original signal', fontsize=15)
    plt.plot(t, x, 'black', color='0.6')

    plt.xlabel("time (ms)", fontsize=15)
    plt.ylabel('Normalized amplitude', fontsize=15)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    plt.legend(['Original signal'], fontsize=12)

    # plt.ylim([-1.2, 1.2])
    # plt.ylim([np.min(x)*1.3, np.max(x)*1.3])

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=100)
        plt.show()


def plot_time_series(x, t, x_, ax, colour, transformation):

    ax.plot(t, x.ravel(), 'black', color='0.4', linewidth=2)
    ax.plot(t, x_.ravel(), colour, ls='--', linewidth=0.75)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    ax.set_xlabel("time (ms)", fontsize=15)
    ax.set_ylabel('Amplitude', fontsize=15)
    ax.set_title(transformation, fontsize=15)

    ax.legend(['Original signal', str(transformation)], fontsize=12, bbox_to_anchor=(1.1, 1.05))

    # ax.ticklabel_format(scilimits=(-3, 3))
    # ax.ticklabel_format(useMathText=True)

    # ax.set_ylim([-0.3, 0.2])
    # ax.set_ylim([np.min(x)*1.5, np.max(x)*1.5])
