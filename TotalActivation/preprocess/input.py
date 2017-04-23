import scipy.io as sio
import numpy as np


def load_matlab_data(d, a=None):
    """
    Loads data and atlas in matlab format

    Inputs:
    d : data (Matlab file with 4D matrix 'data' variable)
    a : atlas (Matlab file with 3D matrix 'atlas' variable) - optional
    """
    data = sio.loadmat(d)['data']
    atlas = sio.loadmat(a)['atlas']
    data = data[np.nonzero(self.atlas * np.ndarray.sum(data, axis=len(data.shape) - 1))].T

    if a is not None:
        return data, atlas
    else:
        return data


def load_nifti(d, a=None):
    """
    Basic function to load NIFTI time-series and flatten them to 2D array

    Inputs:
    d : data (4D NIFTI file)
    a : atlas (3D NIFTI file)
    """
    from nilearn.input_data import NiftiMasker

    data_masker = NiftiMasker(mask_strategy='epi',
                              standardize=self.config['Standardize'],
                              detrend=self.config['Detrend'],
                              high_pass=self.config['Highpass'],
                              low_pass=self.config['Lowpass'],
                              t_r=self.config['TR'])

    data_masker.fit(d)

    if a is not None:
        atlas_masker = NiftiMasker(mask_strategy='background',
                                   standardize=False,
                                   detrend=False)
        atlas_masker.fit(a)

        x1 = data_masker.mask_img_.get_data()
        x2 = atlas_masker.mask_img_.get_data()
        x1 *= x2
        x2 *= x1
        atlas = atlas_masker.transform(a)

    data = data_masker.transform(d)

    if a is not None:
        return data, data_masker, atlas, atlas_masker
    else:
        return data, data_masker


def load_nifti_nomask(d, a=None):
    """
    Basic function to load NIFTI time-series and flatten them to 2D array

    Inputs:
    d : data (4D NIFTI file)
    a : atlas (3D NIFTI file)
    """
    from nilearn.input_data import NiftiMasker
    import nibabel as nib

    mask = np.ones(nib.load(d).shape[0:3])
    maskimg = nib.Nifti1Image(mask, np.eye(4))

    da = nib.load(d)
    data_masker = NiftiMasker(mask_img=maskimg,
                              standardize=False,
                              detrend=False)
    data_masker.fit(da)

    if a is not None:
        at = nib.load(a)

        atlas_masker = NiftiMasker(mask_img=maskimg,
                                   standardize=False,
                                   detrend=False)
        atlas_masker.fit(at)
        atlas = atlas_masker.transform(a)

    data = data_masker.transform(d)

    if a is not None:
        return data, data_masker, atlas, atlas_masker
    else:
        return data, data_masker


def load_text_data(d):
    """
    This file loads a time-by-space data matrix.

    :param d: file in csv format
    :return:
    """

    data = np.genfromtxt(d, delimiter=',')

    return data
