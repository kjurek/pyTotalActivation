from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pywt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def mad(X, axis=0):
    """
    Median absolute deviation

    :param X: Input matrix
    ;param axis: Axis to calculate quantity (default = 0)
    :return: MAD for X along axis
    """

    return np.median(np.abs(X - np.median(X, axis=axis)), axis=axis)


def wiener(X, hrfparam, Lambda, n_vox, n_tp):
    """
    Perform Wiener-based temporal deconvolution.

    :param X: time x voxels matrix
    :param hrfparam: HRF parameters
    :param l: Lambda
    ;param n_vox: number of voxels
    ;param n_tp: number of time points
    :return: Deconvolved time series
    """

    f_num = np.abs(np.fft.fft(hrfparam[0]['num'], n_tp) ** 2)

    f_den = np.abs(np.fft.fft(hrfparam[0]['den'][0], n_tp) * \
                   np.fft.fft(hrfparam[0]['den'][1], n_tp) * \
                   t.hrfparams[0]['den'][-1] * \
                   np.exp(np.arange(1, n_tp + 1) * (t.hrfparams[0]['den'][1].shape[0] - 1) / n_tp)) ** 2

    _, coef = pywt.wavedec(X, 'db3', level=1, axis=0)
    lambda_temp = mad(coef) * Lambda ** 2 * n_tp

    res = np.real(np.fft.ifft(np.fft.fft(X) * (np.repeat(f_den, n_vox).reshape(n_tp, n_vox) / (
        np.repeat(f_den, n_vox).reshape(n_tp, n_vox) + np.kron(f_num, lambda_temp).reshape(n_tp, n_vox))), axis=1))

    return res
