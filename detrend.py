import numpy as np


def detrend_normalize_mat(voxels):
    tcn = np.zeros((voxels.shape[1], voxels.shape[0]))
    for i in range(voxels.shape[0]):
        tcn[:, i] = voxels[i] / np.std(voxels[i])
    return tcn


def detrend_normalize_nii(voxels):
    tcn = np.zeros((voxels.shape[1], voxels.shape[0]))
    for i in range(voxels.shape[0]):
        tcn[:, i] = (voxels[i] - np.mean(voxels[i])) / np.std(voxels[i])
    return tcn