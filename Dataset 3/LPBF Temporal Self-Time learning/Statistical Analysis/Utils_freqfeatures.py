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
import pywt
import scipy.signal as signal

from scipy.stats import kurtosis, skew
from scipy.signal import welch, periodogram
from numpy.fft import fftshift, fft
from scipy.signal import find_peaks
import statistics
from scipy import stats
from collections import Counter
from scipy.stats import entropy
from scipy.signal import hilbert, chirp
from scipy.stats import entropy


def normalize_to_minus_one(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = [((x - min_val) / (max_val - min_val)) * 2 - 1 for x in array]

    normalized_array = np.array(normalized_array)

    return normalized_array


def get_band(band_size, band_max_size):
    band_window = 0
    band = []
    for y in range(band_size):
        band.append(band_window)
        band_window += band_max_size / band_size
    return band


def spectrumpower(psd, band, freqs, band_size):
    length = len(band)
    # print(length)
    Feature_deltapower = []
    Feature_relativepower = []
    for i in range(band_size-1):
        if i <= (len(band)):
            ii = i
            low = band[ii]
            ii = i+1
            high = band[ii]
            idx_delta = np.logical_and(freqs >= low, freqs <= high)
            total_power = sum(psd)
            delta_power = sum(psd[idx_delta])
            delta_rel_power = delta_power / total_power
            Feature_deltapower.append(delta_power)
            Feature_relativepower.append(delta_rel_power)

    return Feature_deltapower, Feature_relativepower


# %%

def function_freq(val, sample_rate, band_size):

    i = 0
    win = 4 * sample_rate
    freqs, psd = periodogram(val, sample_rate, window='hamming')
    band_max_size = 60000
    band = get_band(band_size, band_max_size)

    print(band)

    Feature1, Feature2 = spectrumpower(psd, band, freqs, band_size)
    Feature1 = np.asarray(Feature1)
    Feature2 = np.asarray(Feature2)

    Feature = np.concatenate((Feature1, Feature2))

    if i == 0:
        #     print("--reached")
        size_of_Feature_vectors = int(len(Feature))

        Feature_vectors = np.empty((0, size_of_Feature_vectors))

    # print(label)
    Feature_vectors = np.append(Feature_vectors, [Feature], axis=0)

    return Feature_vectors

# %%


def Freqfunction(data_new, sample_rate, band_size):
    columnsdata = data_new.transpose()
    columns = np.atleast_2d(columnsdata).shape[1]
    featurelist = []
    classlist = []
    rawlist = []

    # for row in loop:
    for k in range(columns):

        val = columnsdata[:, k]
        Feature_vectors = function_freq(val, sample_rate, band_size)
        print(k)
        for item in Feature_vectors:

            featurelist.append(item)

    return featurelist
