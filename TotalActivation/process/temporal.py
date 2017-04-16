from __future__ import absolute_import, division, print_function

import logging
import pywt

import numpy as np

from TotalActivation.filters.filter_boundary import filter_boundary_normal, filter_boundary_transpose

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


# TODO this function needs love
def temporal_TA(y, f_analyze, max_eig, N, Nit, noise_estimate_fin, voxel_ind, lambda_temp, cost_save):
    if noise_estimate_fin is not None and len(noise_estimate_fin) > voxel_ind:
        lambdas_temp_fin = noise_estimate_fin[voxel_ind]
    else:
        lambdas_temp_fin = lambda_temp[voxel_ind - 1]

    if cost_save is not None:
        cost_temp = np.zeros((Nit, 1))
    else:
        cost_temp = None

    noise_estimate = lambda_temp[voxel_ind - 1]
    precision = noise_estimate / 100000.0

    z = np.zeros((N, 1))
    k = 0
    t = 1
    s = np.zeros((N, 1))

    while k < Nit:
        z_l = z
        z0 = filter_boundary_normal(f_analyze, y)
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
            temp = y - lambdas_temp_fin * filter_boundary_transpose(f_analyze, z)
            cost_temp[k] = np.sum(np.power(temp - y, 2)) / 2.0 + lambdas_temp_fin * np.sum(
                np.abs(filter_boundary_normal(f_analyze, temp)))
            noise_estimate_fin = np.sqrt(np.sum(np.power(temp - y, 2.0)) / N)
        else:
            nv_tmp1 = filter_boundary_transpose(f_analyze, z)
            nv_tmp2 = lambdas_temp_fin * nv_tmp1
            noise_estimate_fin = np.sqrt(np.power(np.sum(nv_tmp2), 2.0) / N)

        if np.abs(noise_estimate_fin - noise_estimate) > precision:
            lambdas_temp_fin = lambdas_temp_fin * noise_estimate / noise_estimate_fin
        k += 1

    x = y - lambdas_temp_fin * filter_boundary_transpose(f_analyze, z)
    return x, noise_estimate_fin, lambdas_temp_fin, cost_temp
