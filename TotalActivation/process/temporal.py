from __future__ import absolute_import, division, print_function

import logging
import pywt

import numpy as np

from TotalActivation.filters.filter_boundary import filter_boundary_normal, filter_boundary_transpose

# from TotalActivation.process.utils import mad

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

    f_num = np.abs(np.fft.fft(hrfparam['num'], n_tp) ** 2)

    f_den = np.abs(np.fft.fft(hrfparam['den'][0], n_tp) * \
                   np.fft.fft(hrfparam['den'][1], n_tp) * \
                   hrfparam['den'][-1] * \
                   np.exp(np.arange(1, n_tp + 1) * (hrfparam['den'][1].shape[0] - 1) / n_tp)) ** 2

    _, coef = pywt.wavedec(X, 'db3', level=1, axis=0)
    lambda_temp = mad(coef) * Lambda ** 2 * n_tp

    res = np.real(np.fft.ifft(np.fft.fft(X) * (np.repeat(f_den, n_vox).reshape(n_tp, n_vox) / (
        np.repeat(f_den, n_vox).reshape(n_tp, n_vox) + np.kron(f_num, lambda_temp).reshape(n_tp, n_vox))), axis=1))

    return res


# TODO this function needs love
def temporal_TA(X, f_analyze, max_eig, n_tp, Nit, noise_estimate_fin, l, cost_save, voxels=None):
    if voxels is None:
        _, coef = pywt.wavedec(X, 'db3', level=1, axis=0)
        lambda_temp = mad(coef) * l
        if noise_estimate_fin is not None:
            lambdas_temp_fin = np.atleast_1d(noise_estimate_fin).copy()
        else:
            lambdas_temp_fin = np.atleast_1d(lambda_temp).copy()

        if cost_save is not False:
            cost_temp = np.zeros((Nit, 1))
        else:
            cost_temp = None


    else:
        X = X[:, voxels]
        _, coef = pywt.wavedec(X, 'db3', level=1, axis=0)
        lambda_temp = mad(coef) * l

        if noise_estimate_fin is not None:
            lambdas_temp_fin = np.atleast_1d(noise_estimate_fin[voxels]).copy()
        else:
            lambdas_temp_fin = np.atleast_1d(lambda_temp).copy()

        if cost_save is not False:
            cost_temp = np.zeros((Nit, 1))
        else:
            cost_temp = None

    noise_estimate = np.atleast_1d(lambda_temp).copy()
    noise_estimate = np.minimum(noise_estimate, 0.95)
    precision = noise_estimate / 100000.0

    z = np.zeros_like(X)
    k = 0
    t = 1
    s = np.zeros_like(X)

    while k < Nit:
        z_l = z.copy()
        z0 = filter_boundary_normal(f_analyze, X)
        z1 = 1.0 / (lambdas_temp_fin * max_eig) * z0
        z2 = filter_boundary_transpose(f_analyze, s)
        z3 = filter_boundary_normal(f_analyze, z2)
        z4 = z1 + s
        z = z4 - z3 / max_eig
        z = np.maximum(np.minimum(z, 1), -1)
        t_l = t
        t = (1 + np.sqrt(1.0 + 4.0 * (np.power(t, 2)))) / 2.0
        s = z + (t_l - 1.0) / t * (z - z_l)

        if cost_save is not None:
            temp = X - lambdas_temp_fin * filter_boundary_transpose(f_analyze, z)
            cost_temp = np.sum(np.power(temp - X, 2), axis=0) / 2.0 + lambdas_temp_fin * np.sum(
                np.abs(filter_boundary_normal(f_analyze, temp)), axis=0)
            noise_estimate_fin = np.sqrt(np.sum(np.power(temp - X, 2.0), axis=0) / n_tp)
        else:
            nv_tmp1 = filter_boundary_transpose(f_analyze, z)
            nv_tmp2 = lambdas_temp_fin * nv_tmp1
            noise_estimate_fin = np.sqrt(np.sum(np.power(nv_tmp2, 2.0), axis=0) / n_tp)

        if np.any(np.abs(noise_estimate_fin - noise_estimate) > precision):
            gp = np.where(np.abs(noise_estimate_fin - noise_estimate) > precision)[0]
            lambdas_temp_fin[gp] = lambdas_temp_fin[gp] * noise_estimate[gp] / noise_estimate_fin[gp]
        k += 1

    Y = X - lambdas_temp_fin * filter_boundary_transpose(f_analyze, z)
    return Y  # , noise_estimate_fin, lambdas_temp_fin, cost_temp
