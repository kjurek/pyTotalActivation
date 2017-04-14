from __future__ import absolute_import, division, print_function
import scipy.io as sio
import numpy as np
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

__all__ = ["TotalActivation"]

class TotalActivation(object):
    def __init__(self):
        # Method_time: 'B', 'S' or 'W'
        # Method_space: 'S', 'T', None
        # HRF: 'bold', 'spmhrf'
        self.config = {'Method_time': 'B',
                       'Method_space': None,
                       'HRF': 'bold',
                       'Detrend': True,
                       'Standardize': True,
                       'Highpass': 0.01,
                       'Lowpass': None,
                       'TR': 2,
                       'Lambda': 1 / 0.8095}
        self.data = None
        self.atlas = None

    def load_matlab_data(self, d, a):
        """
        Loads data and atlas in matlab format

        Inputs:
        d : data (Matlab file with 4D matrix 'data' variable)
        a : atlas (Matlab file with 3D matrix 'atlas' variable)
        """
        data = sio.loadmat(d)['data']
        self.atlas = sio.loadmat(a)['atlas']
        self.data = data[np.nonzero(self.atlas * np.ndarray.sum(data, axis=len(data.shape) - 1))]
        logging.debug('self.data.shape={}'.format(self.data.shape))
        logging.debug('self.atlas.shape={}'.format(self.atlas.shape))

    def load_nifti_data(self, d, a):
        """
        Basic function to load NIFTI time-series and flatten them to 2D array

        Inputs:
        d : data (4D NIFTI file)
        a : atlas (3D NIFTI file)
        """
        from nilearn.input_data import NiftiMasker

        atlas_masker = NiftiMasker(mask_strategy='background',
                                   memory="nilearn_cache", memory_level=2,
                                   standardize=False,
                                   detrend=False)

        data_masker = NiftiMasker(mask_strategy='epi',
                                  memory="nilearn_cache", memory_level=2,
                                  standardize=self.config['Standardize'],
                                  detrend=self.config['Detrend'],
                                  high_pass=self.config['Highpass'],
                                  low_pass=self.config['Lowpass'],
                                  t_r=self.config['TR'])

        atlas_masker.fit(a)
        data_masker.fit(d)
        x1 = data_masker.mask_img_.get_data()
        x2 = atlas_masker.mask_img_.get_data()
        x1 *= x2
        x2 *= x1
        self.data = data_masker.transform(d)
        self.atlas = atlas_masker.transform(a)
        logging.debug('self.data.shape={}'.format(self.data.shape))
        logging.debug('self.atlas.shape={}'.format(self.atlas.shape))



if __name__ == '__main__':
    ta = TotalActivation()