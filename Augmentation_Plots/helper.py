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
import matplotlib.pyplot as plt
import random


def plot_d(x, t, x_, colour, save_file='', Transformation=''):
    """
    Plots the original signal and the transformed signal.

    Args:
        x (numpy.ndarray): The original signal.
        t (numpy.ndarray): The time values corresponding to the signal.
        x_ (numpy.ndarray): The transformed signal.
        colour (str): The color of the transformed signal.
        save_file (str, optional): The file path to save the plot. Defaults to ''.
        Transformation (str, optional): The name of the transformation applied to the signal. Defaults to ''.

    Returns:
        None
    """

    fig = plt.figure(figsize=(9, 4))
    plt.title('Original signal -- ' + str(Transformation), fontsize=15)

    plt.plot(t, x.ravel(),  color='black', ls='--',  linewidth=1)
    plt.plot(t, x_.ravel(), colour, ls='--', linewidth=1)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    plt.xlabel("time(s)", fontsize=15)
    plt.ylabel('Normalized amplitude', fontsize=15)

    plt.legend(['Original signal', str(Transformation)],
               fontsize=12, loc='lower left')

    # plt.ylim([-0.2, 0.3])
    # plt.ylim([np.min(x)*1.2, np.max(x)*1.2])

    plt.savefig(save_file, dpi=200)
    plt.show()


def plot1d(x, t, save_file=""):
    """
    Plots a 1-dimensional signal.

    Args:
        x (array-like): The signal values.
        t (array-like): The time values corresponding to the signal.
        save_file (str, optional): The file path to save the plot. Defaults to "".

    Returns:
        None
    """

    plt.figure(figsize=(9, 5))

    plt.title('Original signal', fontsize=15)
    plt.plot(t, x, color='black')

    plt.xlabel("time (ms)", fontsize=15)
    plt.ylabel('Normalized amplitude', fontsize=15)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    plt.legend(['Original signal'], fontsize=12)

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, dpi=100)
        plt.show()


def plot_time_series(x, t, x_, ax, colour, transformation):
    """
    Plots the time series data.

    Args:
        x (numpy.ndarray): The original signal.
        t (numpy.ndarray): The time values.
        x_ (numpy.ndarray): The transformed signal.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        colour (str): The color of the transformed signal plot.
        transformation (str): The name of the transformation.

    Returns:
        None
    """

    ax.plot(t, x.ravel(), color='black', linewidth=0.75)
    ax.plot(t, x_.ravel(), colour, ls='--', linewidth=1.5)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    ax.set_xlabel("time (s)", fontsize=15)
    ax.set_ylabel('Normalized signal', fontsize=15)
    ax.set_title(transformation, fontsize=15)

    # ax.legend(['Original signal', str(transformation)],
    #           fontsize=12, bbox_to_anchor=(1.1, 1.05))
    ax.legend(['Original signal', str(transformation)], loc='lower right',
              fontsize=12)

    # ax.ticklabel_format(scilimits=(-3, 3))
    # ax.ticklabel_format(useMathText=True)

    # ax.set_ylim([-0.3, 0.2])
    # ax.set_ylim([np.min(x)*1.5, np.max(x)*1.5])
