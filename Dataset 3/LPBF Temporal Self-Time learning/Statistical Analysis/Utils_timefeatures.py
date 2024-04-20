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


def Zerocross(a):
    zero_crossings = np.where(np.diff(np.signbit(a)))[0]
    cross = zero_crossings.size
    #print (cross)
    return cross


# %%


def autopeaks(psd):
    import scipy.signal
    indexes, value = scipy.signal.find_peaks(psd, height=0, distance=None)
    a = value['peak_heights']
    sorted_list = sorted(a, reverse=True)
    b = sorted_list[0:4]
    b_size = int(len(b))

    if b_size < 4:
        # replace missing values with zeros
        b_missing = 4 - b_size
        for x in range(b_missing):
            b.append(0)
    return b


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]


def get_autocorr_values(y_values):
    autocorr_values = autocorr(y_values)
    peaks = autopeaks(autocorr_values)
    peaks = np.asarray(peaks)
    return peaks


# %%

def function_time(val, window):

    # data_new=data_new.transpose()

    Feature_vectors = []
    signal_chop = np.split(val, window)
    for i in range(window):

        signal_window = signal_chop[i]

        # minimum
        Feature1 = signal_window.min()
        # maximum
        Feature2 = signal_window.max()
        # difference
        Feature3 = Feature2+Feature1
        # difference
        Feature4 = Feature2+abs(Feature1)
        # RMS
        Feature5 = np.sqrt(np.mean(signal_window**2))
        # print(Feature5)
        # STD
        Feature6 = statistics.stdev(signal_window)
        # Variance
        Feature7 = statistics.variance(signal_window)
        # Skewness
        Feature8 = skew(signal_window)
        # Kurtosis
        Feature9 = kurtosis(signal_window)
        # Mean
        Feature10 = statistics.mean(signal_window)
        # Harmonic Mean
        Feature11 = statistics.harmonic_mean(abs(signal_window))
        # Median
        Feature12 = statistics.median(signal_window)
        # Median_1
        Feature13 = Feature12-Feature11
        # Zerocrossing
        Feature14 = Zerocross(signal_window)
        # Mean Absolute Deviation
        Feature15 = stats.median_abs_deviation(signal_window)
        # Absolute Mean
        Feature16 = statistics.mean(abs(signal_window))
        # Absolute RMS
        Feature17 = np.sqrt(np.mean(abs(signal_window)**2))
        # Absolute Max
        Feature18 = max(abs(signal_window))
        # Absolute Min
        Feature19 = min(abs(signal_window))
        # Absolute Mean -  Mean
        Feature20 = ((abs(signal_window)).mean())-(signal_window.mean())
        # difference+Median
        Feature21 = Feature3+Feature12
        # Crest factor - peak/ rms
        Feature22 = Feature2/Feature5
        # Auto correlation 4 peaks
        Feature23 = get_autocorr_values(signal_window)

        Feature = [Feature1, Feature2, Feature3, Feature4, Feature5,
                   Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15,
                   Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22]

        Feature_1 = np.concatenate((Feature, Feature23))

        # Create the size of numpy array, by checking the size of "Feature_1" and creating "Feature_vectors" with the required shape on first run
        if i == 0:
            #     print("--reached")
            size_of_Feature_vectors = int(len(Feature_1))
            size_of_dataset = int(len(signal_window))
            Feature_vectors = np.empty((0, size_of_Feature_vectors))

        Feature_vectors = np.append(Feature_vectors, [Feature_1], axis=0)

    return Feature_vectors

# %%


def Timefunction(data_new, windowsize):
    columnsdata = data_new.transpose()
    columns = np.atleast_2d(columnsdata).shape[1]
    featurelist = []

    # for row in loop:
    for k in range(columns):

        val = columnsdata[:, k]
        totaldatapoints = val.size
        window = round(totaldatapoints/windowsize)

        # print("Before function",val.shape)

        Feature_vectors = function_time(val, window)

        print(k)

        for item in Feature_vectors:

            featurelist.append(item)

    return featurelist

# %%
