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
from tqdm import tqdm


def cut_piece2C(ts, perc=.1):
    """
    Cut two pieces from a given time series and return them along with a label.
    Args:
        ts (numpy.ndarray): The input time series.
        perc (float, optional): The percentage or length of the time series to be cut. Defaults to 0.1.
    Returns:
        tuple: A tuple containing two numpy arrays representing the cut pieces of the time series and a label.
    Comment:
        The function randomly selects two start and end points within the time series and cuts two pieces of length
        'perc' or percentage of the time series. It then checks the distance between the start points and assigns a
        label of 0 if the distance is less than half of the length of the time series divided by 2, otherwise assigns
        a label of 1.
    """

    seq_len = ts.shape[0]
    win_class = seq_len/(2*2)

    if perc < 1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len-win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1-start2) < (win_class):
        label = 0
    else:
        label = 1

    return ts[start1:end1, ...], ts[start2:end2, ...], label


def cut_piece3C(ts, perc=.1):
    """
    Cut two pieces from a given time series and return them along with a label.
    Args:
        ts (numpy.ndarray): The input time series.
        perc (float, optional): The percentage or length of the time series to be cut. Defaults to 0.1.
    Returns:
        tuple: A tuple containing two pieces of the time series and a label.
            - The first piece of the time series.
            - The second piece of the time series.
            - The label indicating the relationship between the two pieces.
    Comment: The function randomly selects two segments of the time series based on the given percentage or length.
    The label is determined based on the distance between the start points of the two segments.
    """

    seq_len = ts.shape[0]
    win_class = seq_len/(2*3)

    if perc < 1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len-win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1-start2) < (win_class):
        label = 0
    elif abs(start1-start2) < (2*win_class):
        label = 1
    else:
        label = 2

    return ts[start1:end1, ...], ts[start2:end2, ...], label


def slidewindow(ts, horizon=.2, stride=0.2):
    """
    Slides a window over a time series and extracts input and output sequences.
    Args:
        ts (numpy.ndarray): The input time series.
        horizon (float, optional): The length of the output sequence as a fraction of the total length of the time series. Defaults to 0.2.
        stride (float, optional): The stride length as a fraction of the total length of the time series. Defaults to 0.2.
    Returns:
        tuple: A tuple containing two numpy arrays. The first array contains the input sequences, and the second array contains the corresponding output sequences.
    Comment: The function slides a window over the input time series and extracts input and output sequences based on the specified horizon and stride lengths.
    """

    xf = []
    yf = []
    for i in range(0, ts.shape[0], int(stride * ts.shape[0])):
        horizon1 = int(horizon * ts.shape[0])
        if (i + horizon1 + horizon1 <= ts.shape[0]):
            xf.append(ts[i:i + horizon1, 0])
            yf.append(ts[i + horizon1:i + horizon1 + horizon1, 0])

    xf = np.asarray(xf)
    yf = np.asarray(yf)

    return xf, yf


def cutout(ts, perc=.1):
    """
    Applies cutout augmentation to a given time series.
    Args:
        ts (numpy.ndarray): The input time series.
        perc (float, optional): The percentage of the time series to be cut out. Defaults to 0.1.
    Returns:
        numpy.ndarray: The augmented time series with cutout applied.
    Note:
        - The input time series `ts` should be a numpy array.
        - The `perc` parameter determines the percentage of the time series to be cut out.
        - The function modifies the input time series in-place and returns the modified time series.
    """

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


def jitter(x, sigma=0.03):
    """
    Applies jitter augmentation to the input data.
    Args:
        x (ndarray): The input data.
        sigma (float, optional): The standard deviation of the normal distribution used for jittering. 
            Defaults to 0.03.
    Returns:
        ndarray: The augmented data with jitter applied.
    Note:
        This function applies jitter augmentation to the input data by adding random noise from a normal distribution
        with mean 0 and standard deviation sigma to each element of the input array.
    Reference:
        - "Improved Regularization of Convolutional Neural Networks with Cutout" by DeVries and Taylor (2017)
          (https://arxiv.org/pdf/1708.04552.pdf)
    """

    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    """
    Applies scaling augmentation to the input data.
    Args:
        x (numpy.ndarray): The input data to be scaled.
        sigma (float, optional): The standard deviation of the scaling factor. Defaults to 0.1.
    Returns:
        numpy.ndarray: The scaled input data.
    Note:
        The scaling factor is generated from a normal distribution with mean 1 and standard deviation sigma.
    """

    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(1, x.shape[1]))
    x_ = np.multiply(x, factor[:, :])

    return x_


def magnitude_warp(x, sigma=0.2, knot=4, plot=False):
    """
    Apply magnitude warp augmentation to the input data.
    Parameters:
        x (numpy.ndarray): The input data to be augmented.
        sigma (float): The standard deviation of the random warps. Default is 0.2.
        knot (int): The number of knots for the warping function. Default is 4.
        plot (bool): Whether to plot the warping function. Default is False.
    Returns:
        numpy.ndarray: The augmented data after applying magnitude warp.
    # Example usage:
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    augmented_x = magnitude_warp(x, sigma=0.3, knot=5, plot=True)
    """

    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(1, knot + 2, x.shape[1]))
    warp_steps = (
        np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1., num=knot + 2))).T

    li = []
    for dim in range(x.shape[1]):
        li.append(CubicSpline(warp_steps[:, dim],
                  random_warps[0, :, dim])(orig_steps))
    warper = np.array(li).T

    x_ = x * warper

    return x_


def time_warp(x, sigma=0.2, knot=4, plot=False):
    """
    Apply time warping augmentation to the input data.
    Parameters:
    - x (numpy.ndarray): The input data array of shape (n_samples, n_features).
    - sigma (float): The standard deviation of the random warping factors. Default is 0.2.
    - knot (int): The number of knots used for warping. Default is 4.
    - plot (bool): Whether to plot the warped data. Default is False.
    Returns:
    - numpy.ndarray: The warped data array of the same shape as the input data.
    Note:
    - The input data should be a 2-dimensional numpy array.
    - The output data will have the same shape as the input data.
    - The time warping is performed using cubic spline interpolation.
    - The random warping factors are generated from a normal distribution with mean 1.0 and standard deviation sigma.
    - The warping steps are determined by evenly spaced knots between 0 and n_samples-1.
    - The warped data is obtained by interpolating the original data using the warped time steps.
    Example usage:
    >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> warped_x = time_warp(x, sigma=0.2, knot=4, plot=True)
    """
    # Implementation details:
    # - The original time steps are generated using np.arange.
    # - The random warping factors are generated using np.random.normal.
    # - The warping steps are determined by evenly spaced knots between 0 and n_samples-1.
    # - The time warping is performed using CubicSpline interpolation.
    # - The warped data is obtained by interpolating the original data using the warped time steps.
    # - The scale factor is calculated to ensure that the last warped time step is equal to n_samples-1.
    # - The np.interp function is used to perform the interpolation.
    # - The np.clip function is used to ensure that the warped time steps are within the valid range.
    # - The .T attribute is used to transpose the interpolated data.
    # Add any additional notes or explanations here.

    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(1, knot + 2, x.shape[1]))
    warp_steps = (
        np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for dim in range(x.shape[1]):
        time_warp = CubicSpline(warp_steps[:, dim],
                                warp_steps[:, dim] * random_warps[0, :, dim])(orig_steps)
        scale = (x.shape[0] - 1) / time_warp[-1]
        ret[:, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[0] - 1),
                                x[:, dim]).T

    return ret


def window_slice(x, reduce_ratio=0.9):
    """
    Slice the input array `x` along the time axis by reducing its length based on the `reduce_ratio`.
    Args:
        x (ndarray): The input array of shape (n_samples, n_features) representing the data.
        reduce_ratio (float, optional): The ratio by which to reduce the length of `x`. Defaults to 0.9.
    Returns:
        ndarray: The sliced array with reduced length along the time axis.
    Note:
        The input array `x` should have a shape of (n_samples, n_features).
        The output array will have the same shape as the input array, but with a reduced length along the time axis.
    """
    # Input type: ndarray
    # Output type: ndarray

    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio * x.shape[0]).astype(int)
    if target_len >= x.shape[0]:
        return x
    starts = np.random.randint(
        low=0, high=x.shape[0] - target_len, size=(1)).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for dim in range(x.shape[1]):
        ret[:, dim] = np.interp(np.linspace(0, target_len, num=x.shape[0]), np.arange(target_len),
                                x[starts[0]:ends[0], dim]).T
    return ret


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    """
    Apply window warp augmentation to the input data.
    Args:
        x (numpy.ndarray): The input data array of shape (n_samples, n_features).
        window_ratio (float, optional): The ratio of the window size to the total data size. Defaults to 0.1.
        scales (list, optional): The list of scales to randomly choose from for warping. Defaults to [0.5, 2.].
    Returns:
        numpy.ndarray: The augmented data array of shape (n_samples, n_features).
    Note:
        The function applies window warp augmentation to the input data by randomly warping a window segment within the data.
        The window size is determined by the window_ratio parameter, and the scales parameter determines the range of scaling factors for warping.
    """
    # Implementation details:
    # - Randomly choose a scale from the scales list
    # - Calculate the size of the window based on the window_ratio
    # - Generate an array of window steps
    # - Randomly choose a starting position for the window within the data
    # - Calculate the ending position of the window
    # - Create an empty array to store the augmented data
    # - Iterate over each dimension of the input data
    # - Extract the segments before, within, and after the window
    # - Warp the window segment using linear interpolation
    # - Concatenate the segments to form the augmented data
    # - Return the augmented data array

    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, 1)
    warp_size = np.ceil(window_ratio * x.shape[0]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(
        low=1, high=x.shape[0] - warp_size - 1, size=(1)).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    pat = x
    for dim in range(x.shape[1]):
        start_seg = pat[:window_starts[0], dim]
        window_seg = np.interp(np.linspace(0, warp_size - 1,
                                           num=int(warp_size * warp_scales[0])), window_steps,
                               pat[window_starts[0]:window_ends[0], dim])
        end_seg = pat[window_ends[0]:, dim]
        warped = np.concatenate((start_seg, window_seg, end_seg))
        ret[:, dim] = np.interp(np.arange(x.shape[0]), np.linspace(0, x.shape[0] - 1., num=warped.size),
                                warped).T
    return ret
