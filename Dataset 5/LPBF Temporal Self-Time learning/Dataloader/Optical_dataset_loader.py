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

# %%
# Normalize the data to [-1, 1]
def normalize_to_minus_one(array):
    """
    Normalize an array to the range of -1 to 1.

    Args:
        array (numpy.ndarray): The input array to be normalized.

    Returns:
        numpy.ndarray: The normalized array.

    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = [((x - min_val) / (max_val - min_val)) * 2 - 1 for x in array]
    normalized_array = np.array(normalized_array)

    return normalized_array


def load_LPBF(path, dataset_name, dataset_label):
    """
    Load LPBF dataset from the given path.

    Args:
        path (str): The path to the dataset.
        dataset_name (str): The name of the raw data file.
        dataset_label (str): The name of the class labels file.

    Returns:
        tuple: A tuple containing the following elements:
            - x_train (ndarray): The training data.
            - y_train (ndarray): The training labels.
            - x_val (ndarray): The validation data.
            - y_val (ndarray): The validation labels.
            - x_test (ndarray): The test data.
            - y_test (ndarray): The test labels.
            - nb_class (ndarray): The unique classes in the dataset.
    """
    # Function code here

    ##################
    # load raw data
    ##################

    print("Dataset path...", path)
    print("Dataset name...", dataset_name)

    rawspace = np.load("{}/{}".format(path, dataset_name))
    classspace = np.load("{}/{}".format(path, dataset_label))

    rawspace = normalize_to_minus_one(rawspace)
    rawspace = pd.DataFrame(rawspace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    data = pd.concat([rawspace, classspace], axis=1)

    print("Respective windows per category", data.Categorical.value_counts())
    minval = min(data.Categorical.value_counts())

    # minval=np.round(minval,decimals=-3)
    print("windows of the class: ", minval)

    data_1 = pd.concat([data[data.Categorical == cat].head(minval)
                       for cat in data.Categorical.unique()])
    print("Balanced dataset: ", data_1.Categorical.value_counts())

    data = data_1.iloc[:, :-1]
    label = data_1.iloc[:, -1]

    x = data.to_numpy()
    y = label.to_numpy()

    input_shape = x.shape[1]
    nb_class = np.unique(y)

    print("Unique classes in the dataset  ", nb_class)
    ############################################
    # Combine all train and test data for resample
    ############################################

    ts_idx = list(range(x.shape[0]))
    np.random.shuffle(ts_idx)
    x_all = x[ts_idx]
    y_all = y[ts_idx]

    Features_train_max = np.max(x_all)
    Features_train_min = np.min(x_all)
    label_idxs = np.unique(y_all)

    test_idx = []
    val_idx = []
    train_idx = []

    for idx in label_idxs:
        target = list(np.where(y_all == idx)[0])
        nb_samp = int(len(target))
        test_idx += target[:int(nb_samp * 0.30)]
        val_idx += target[int(nb_samp * 0.30):int(nb_samp * 0.40)]
        train_idx += target[int(nb_samp * 0.60):]

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]

    x_val = x_all[val_idx]
    y_val = y_all[val_idx]

    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    print("[Stat] Whole dataset: mean={}, std={}".format(np.mean(x_all), np.std(x_all)))
    print("[Stat-normalize] Train class: mean={}, std={}".format(np.mean(x_train), np.std(x_train)))
    print("[Stat-normalize] Val class: mean={}, std={}".format(np.mean(x_val), np.std(x_val)))
    print("[Stat-normalize] Test class: mean={}, std={}".format(np.mean(x_test), np.std(x_test)))

    # reshaping the data
    x_test = x_test.reshape((-1, input_shape, 1))
    x_val = x_val.reshape((-1, input_shape, 1))
    x_train = x_train.reshape((-1, input_shape, 1))

    print("Train:{}, Test:{},Val:{} ,Class:{}".format(
        x_train.shape, x_test.shape, x_val.shape, nb_class))

    return x_train, y_train, x_val, y_val, x_test, y_test, nb_class
