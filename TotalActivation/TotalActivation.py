from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import scipy.io as sio
import time

import joblib
from joblib import Parallel, delayed


from TotalActivation.filters import hrf
from TotalActivation.process.temporal import wiener
from TotalActivation.process.spatial import tikhonov
from TotalActivation.process.utils import parallel_temporalTA

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
        self.deconvolved_ = None
        self.n_voxels = 0
        self.n_tp = 0
        self.cost_save = False
        self.masking = False
        self.n_jobs = -2
        self.n_iter = 5

        self._get_hrf_parameters()

    def _get_hrf_parameters(self):
        """
        Prepares a field with HRF parameters

        :return:
        """

        if self.config['HRF'] == 'bold':
            a, psi = hrf.bold_parameters()
        elif self.config['HRF'] == 'spmhrf':
            a, psi = hrf.spmhrf_parameters()
        else:
            raise ValueError("HRF must be either bold or spmhrf")

        if self.config['Method_time'] == 'B':
            self.hrfparams = hrf.block_filter(a, psi, self.config['TR'])
            self.t_iter = 500
        elif self.config['Method_time'] == 'S':
            self.hrfparams = hrf.spike_filter(a, psi, self.config['TR'])
            self.t_iter = 200
        elif self.config['Method_time'] == 'W':
            self.hrfparams = hrf.block_filter(a, psi, self.config['TR'])
            self.t_iter = 1
        else:
            raise ValueError('Method_time has to be B, S or W')

    def _temporal(self, d):
        """
        Temporal regularization.
        """

        if self.config['Method_time'] is 'B' or self.config['Method_time'] is 'S':
            # _, coef = pywt.wavedec(d, 'db3', level=1, axis=0)
            # lambda_temp = mad(coef) * self.config['Lambda']
            voxels = np.arange(self.n_voxels)
            tempmem = np.memmap('temp.mmap', dtype=float, shape=(self.n_tp, self.n_voxels), mode="w+")

            if self.n_jobs < 0:
                n_splits = joblib.cpu_count() + self.n_jobs + 1
            else:
                n_splits = self.n_jobs

            Parallel(n_jobs=self.n_jobs)(
                delayed(parallel_temporalTA)(d, tempmem, x, self.config['Lambda'], self.hrfparams[0], self.hrfparams[2],
                                             self.n_tp, self.t_iter, self.cost_save)
                for x in np.array_split(voxels, n_splits))

            self.deconvolved_ = tempmem

        elif self.config['Method_time'] is 'W':
            self.deconvolved_ = wiener(d, self.hrfparams[0], self.config['Lambda'], self.n_voxels, self.n_tp)
        else:
            print("Wrong temporal deconvolution method; must be B, S or W")

    def _spatial(self, d, a):
        """
        Spatial regularization.
        """

        if self.config['Method_space'] is 'T':
            self.deconvolved_ = tikhonov(d, a, self.data_masker, iter=self.s_iter)
        else:
            print("This spatial regularization method is not yet implemented")

    def _deconvolve(self):
        """
        Main control function for deconvolution

        :return:
        """

        if self.config['Method_space'] == None:
            print("Temporal regularization...")
            self.t_iter *= 5
            t0 = time.time()
            self._temporal(self.data)
            print("Done in %d seconds!" % (time.time() - t0))
        elif self.config['Method_space'] == 'S':
            print("Structured sparsity spatial regularization not yet implemented")
        elif self.config['Method_space'] == 'T':
            self.s_iter = 100
            TC_OUT = np.zeros_like(self.data)
            xT = np.zeros_like(self.data)
            xS = np.zeros_like(self.data)
            t0 = time.time()
            k = 0
            while k < self.n_iter:


                print("Iteration %d of %d" % (k + 1, self.n_iter))
                print("Temporal...")
                self._temporal(TC_OUT - xT + self.data)
                xT += self.deconvolved_ - TC_OUT
                print("Spatial...")
                self._spatial(TC_OUT, TC_OUT - xS + self.data)
                xS += self.deconvolved_ - TC_OUT
                TC_OUT = 0.5 * xT + 0.5 * xS
                k += 1
            self.deconvolved_ = TC_OUT
            print("Done in %d seconds!" % (time.time() - t0))
        else:
            raise ValueError("Method_space must be S, T or None")

    def load_matlab_data(self, d, a):
        """
        Loads data and atlas in matlab format

        Inputs:
        d : data (Matlab file with 4D matrix 'data' variable)
        a : atlas (Matlab file with 3D matrix 'atlas' variable)
        """
        data = sio.loadmat(d)['data']
        self.atlas = sio.loadmat(a)['atlas']
        self.data = data[np.nonzero(self.atlas * np.ndarray.sum(data, axis=len(data.shape) - 1))].T
        self.n_voxels = self.data.shape[1]
        self.n_tp = self.data.shape[0]
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
        import nibabel as nib

        if self.masking is True:
            self.atlas_masker = NiftiMasker(mask_strategy='background',
                                            standardize=False,
                                            detrend=False)

            self.data_masker = NiftiMasker(mask_strategy='epi',
                                           standardize=self.config['Standardize'],
                                           detrend=self.config['Detrend'],
                                           high_pass=self.config['Highpass'],
                                           low_pass=self.config['Lowpass'],
                                           t_r=self.config['TR'])

            self.atlas_masker.fit(a)
            self.data_masker.fit(d)

        else:
            mask = np.ones(nib.load(d).shape[0:3])
            maskimg = nib.Nifti1Image(mask, np.eye(4))
            da = nib.load(d)
            at = nib.load(a)

            self.atlas_masker = NiftiMasker(mask_img=maskimg,
                                            standardize=False,
                                            detrend=False)

            self.data_masker = NiftiMasker(mask_img=maskimg,
                                           standardize=False,
                                           detrend=False)

            self.atlas_masker.fit(at)
            self.data_masker.fit(da)

        x1 = self.data_masker.mask_img_.get_data()
        x2 = self.atlas_masker.mask_img_.get_data()
        x1 *= x2
        x2 *= x1
        self.data = self.data_masker.transform(d)
        self.atlas = self.atlas_masker.transform(a)
        self.n_voxels = self.data.shape[1]
        self.n_tp = self.data.shape[0]
        logging.debug('self.data.shape={}'.format(self.data.shape))
        logging.debug('self.atlas.shape={}'.format(self.atlas.shape))

    def load_text_data(self, d):
        """
        This file loads a time-by-space data matrix.

        :param d: file in csv format
        :return:
        """

        self.data = np.genfromtxt(d, delimiter=',')
        self.n_voxels = self.data.shape[1]
        self.n_tp = self.data.shape[0]


if __name__ == '__main__':
    ta = TotalActivation()
