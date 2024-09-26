# -*- coding: utf-8 -*-
"""
@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch,vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work
"Pyrometry-based in-situ Layer Thickness Identification via Vector-length Aware Self-Supervised Learning"

@any reuse of this code should be authorized by the code author
"""
# %%
# Libraries to import
import numpy as np
import pandas as pd


def normalize_to_minus_one(array):
    """
    Normalize the given array to the range of -1 to 1.

    Parameters:
    array (numpy.ndarray): The input array to be normalized.

    Returns:
    numpy.ndarray: The normalized array with values ranging from -1 to 1.
    """

    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = [
        ((x - min_val) / (max_val - min_val)) * 2 - 1 for x in array]

    normalized_array = np.array(normalized_array)

    return normalized_array


def load_LPBF(path, dataset_name, dataset_label):
    """
    Load LPBF dataset from the given path with the specified dataset name and label.
    Args:
        path (str): The path to the dataset.
        dataset_name (str): The name of the dataset file.
        dataset_label (str): The name of the label file.
    Returns:
        tuple: A tuple containing the following elements:
            - x_train (ndarray): The training data.
            - y_train (ndarray): The training labels.
            - x_val (ndarray): The validation data.
            - y_val (ndarray): The validation labels.
            - x_test (ndarray): The test data.
            - y_test (ndarray): The test labels.
            - nb_class (ndarray): The unique classes in the dataset.
    Note:
        - The dataset files should be in the numpy format.
        - The dataset and label files should have the same number of samples.
        - The dataset and label files should be in the same order.
    """

    print("dataset_path...", path)
    print("dataset_name...", dataset_name)

    Featurespace = np.load("{}/{}".format(path, dataset_name))
    classspace = np.load("{}/{}".format(path, dataset_label))

    Featurespace = pd.DataFrame(Featurespace)
    classspace = pd.DataFrame(classspace)
    classspace.columns = ['Categorical']
    data = pd.concat([Featurespace, classspace], axis=1)

    print("respective windows", data.Categorical.value_counts())
    minval = min(data.Categorical.value_counts())

    if minval >= 6000:
        minval = 6000
    else:
        minval = minval

    print("windows of the class: ", minval)

    data_1 = pd.concat([data[data.Categorical == cat].head(minval)
                       for cat in data.Categorical.unique()])
    print("The dataset is well balanced: ", data_1.Categorical.value_counts())

    data = data_1.iloc[:, :-1]
    label = data_1.iloc[:, -1]

    x = data.to_numpy()
    y = label.to_numpy()

    input_shape = x.shape[1]

    nb_class = np.unique(y)
    print(
        "Unique classes in the dataset [LoF, Conduction, Keyhole] ", len(nb_class))
    ############################################
    # Combine all train and test data for resample
    ############################################

    ts_idx = list(range(x.shape[0]))
    np.random.shuffle(ts_idx)
    x_all = x[ts_idx]
    y_all = y[ts_idx]

    label_idxs = np.unique(y_all)

    test_idx = []
    val_idx = []
    train_idx = []
    for idx in label_idxs:
        target = list(np.where(y_all == idx)[0])
        nb_samp = int(len(target))
        test_idx += target[:int(nb_samp * 0.25)]
        val_idx += target[int(nb_samp * 0.25):int(nb_samp * 0.5)]
        train_idx += target[int(nb_samp * 0.5):]

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]

    x_val = x_all[val_idx]
    y_val = y_all[val_idx]

    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    print("[Stat] Whole dataset: mean={}, std={}".format(
        np.mean(x_all), np.std(x_all)))
    print("[Stat] Train class: mean={}, std={}".format(
        np.mean(x_train), np.std(x_train)))
    print("[Stat] Val class: mean={}, std={}".format(
        np.mean(x_val), np.std(x_val)))
    print("[Stat] Test class: mean={}, std={}".format(
        np.mean(x_test), np.std(x_test)))

    # x_train=normalize(x_train,Features_train_max,Features_train_min)
    # x_val=normalize(x_val,Features_train_max,Features_train_min)
    # x_test=normalize(x_test,Features_train_max,Features_train_min)

    x_train = normalize_to_minus_one(x_train)
    x_val = normalize_to_minus_one(x_val)
    x_test = normalize_to_minus_one(x_test)

    print("Train:{}, Test:{}, Class:{}".format(
        x_train.shape, x_test.shape, nb_class))

    print("[Stat-normalize] Train class: mean={}, std={}".format(np.mean(x_train), np.std(x_train)))
    print("[Stat-normalize] Val class: mean={}, std={}".format(np.mean(x_val), np.std(x_val)))
    print("[Stat-normalize] Test class: mean={}, std={}".format(np.mean(x_test), np.std(x_test)))

    # Process data
    x_test = x_test.reshape((-1, input_shape, 1))
    x_val = x_val.reshape((-1, input_shape, 1))
    x_train = x_train.reshape((-1, input_shape, 1))

    print("Train:{}, Test:{}, Class:{}".format(
        x_train.shape, x_test.shape, nb_class))

    return x_train, y_train, x_val, y_val, x_test, y_test, nb_class


def jitter(x, sigma=0.2):
    """
    Applies jitter to the input array.
    Parameters:
    - x (ndarray): The input array.
    - sigma (float): The standard deviation of the normal distribution used for jittering. Default is 0.2.
    Returns:
    - ndarray: The jittered array.
    Note:
    - The input array `x` should be a numpy ndarray.
    - The output array will have the same shape as the input array.
    """
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.3):
    """
    Apply scaling augmentation to the input data.
    Parameters:
    x (numpy.ndarray): The input data to be scaled.
    sigma (float, optional): The standard deviation of the normal distribution used to generate the scaling factor. Default is 0.3.
    Returns:
    numpy.ndarray: The scaled data.
    Note:
    The scaling factor is generated from a normal distribution with mean 1 and standard deviation sigma.
    """

    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(
        loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def rotation(x):
    """
    Apply rotation augmentation to the input data.
    Parameters:
    x (ndarray): Input data of shape (batch_size, height, width, channels).
    Returns:
    ndarray: Augmented data with rotation applied, of shape (batch_size, height, width, channels).
    """

    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def cutout(ts, perc=.1):
    """
    Applies cutout augmentation to a given time series.
    Args:
        ts (numpy.ndarray): The input time series.
        perc (float, optional): The percentage of the time series to be cut out. Defaults to 0.1.
    Returns:
        numpy.ndarray: The augmented time series with a portion of values set to 0.
    Example:
        >>> ts = np.array([1, 2, 3, 4, 5])
        >>> cutout(ts, perc=0.2)
        array([0, 0, 3, 4, 5])
    """
    # Implementation details:
    # - The function randomly selects a window of the time series based on the given percentage.
    # - The selected window is set to 0 in the augmented time series.
    # - The augmented time series is returned.

    seq_len = ts.shape[0]
    new_ts = ts.copy()
    win_len = int(perc * seq_len)
    start = np.random.randint(0, seq_len-win_len-1)
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)
    # print("[INFO] start={}, end={}".format(start, end))
    new_ts[start:end, ...] = 0
    # return new_ts, ts[start:end, ...]
    return new_ts


def permutation(x, max_segments=5, seg_mode="equal"):
    """
    Permutes the elements of each pattern in the input array `x` based on the specified segmentation mode.
    Parameters:
        x (ndarray): Input array of shape (n_samples, n_steps) containing patterns.
        max_segments (int, optional): Maximum number of segments to split each pattern into. Defaults to 5.
        seg_mode (str, optional): Segmentation mode. Can be "equal" or "random". Defaults to "equal".
    Returns:
        ndarray: Permuted array of the same shape as `x`, where each pattern has its elements permuted based on the segmentation mode.
    # Example usage:
    x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    permuted_x = permutation(x, max_segments=3, seg_mode="random")
    print(permuted_x)
    # Output: [[1 2 3 4 5] [6 8 7 9 10]]
    """

    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(
                    x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def magnitude_warp(x, sigma=0.3, knot=4):
    """
    Apply magnitude warping to the input array.
    Parameters:
    - x (ndarray): Input array of shape (n_samples, n_steps, n_features).
    - sigma (float): Standard deviation of the normal distribution used for warping. Default is 0.3.
    - knot (int): Number of knots used for warping. Default is 4.
    Returns:
    - ndarray: Warped array of the same shape as the input array.
    Note:
    - The input array should have shape (n_samples, n_steps, n_features).
    - The output array will have the same shape as the input array.
    """
    # Implementation of magnitude warping
    ...

    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) *
                  (np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])
                          (orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret


def time_warp(x, sigma=0.3, knot=4):
    """
    Apply time warping augmentation to the input data.
    Parameters:
    - x (numpy.ndarray): Input data of shape (batch_size, sequence_length, num_features).
    - sigma (float): Standard deviation of the random warping factor. Default is 0.3.
    - knot (int): Number of knots for the cubic spline interpolation. Default is 4.
    Returns:
    - numpy.ndarray: Augmented data of the same shape as the input.
    # Example usage:
    x = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    augmented_x = time_warp(x, sigma=0.5, knot=3)
    """

    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) *
                  (np.linspace(0, x.shape[1]-1., num=knot+2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim]
                                    * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(
                scale*time_warp, 0, x.shape[1]-1), pat[:, dim]).T
    return ret


def window_slice(x, reduce_ratio=0.7):
    """
    Slice the input array `x` along the time dimension with a given reduce ratio.
    Args:
        x (ndarray): Input array of shape (batch_size, time_steps, features).
        reduce_ratio (float, optional): The ratio by which to reduce the length of the time dimension. Defaults to 0.7.
    Returns:
        ndarray: Sliced array of the same shape as `x` with reduced time dimension.
    Note:
        The input array `x` should have a shape of (batch_size, time_steps, features).
        The output array will have the same shape as the input array, but with the time dimension reduced by the given ratio.
    """

    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(
        low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(
                target_len), pat[starts[i]:ends[i], dim]).T
    return ret


def window_warp(x, window_ratio=0.3, scales=[0.5, 2.]):
    """
    Apply window warp augmentation to the input data.
    Args:
        x (ndarray): Input data of shape (batch_size, sequence_length, num_features).
        window_ratio (float, optional): Ratio of the window size to the sequence length. Defaults to 0.3.
        scales (list, optional): List of scales to randomly choose from for warping. Defaults to [0.5, 2.].
    Returns:
        ndarray: Augmented data of the same shape as the input.
    Note:
        The input data should be a 3-dimensional numpy array, where the first dimension represents the batch size,
        the second dimension represents the sequence length, and the third dimension represents the number of features.
    """

    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(
        low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i], dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size *
                                   warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i], dim])
            end_seg = pat[window_ends[i]:, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(
                0, x.shape[1]-1., num=warped.size), warped).T
    return ret
