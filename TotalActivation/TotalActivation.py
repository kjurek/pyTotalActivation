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
from TotalActivation.preprocess.input import load_nifti, load_nifti_nomask, load_matlab_data, load_text_data
from TotalActivation.process.utils import parallel_temporalTA

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

__all__ = ["TotalActivation"]


class TotalActivation(object):
    def __init__(self, method_time='B', method_space='S', hrf='bold', Lambda=1 / 0.8095, cost_save=False):
        # Method_time: 'B', 'S' or 'W'
        # Method_space: 'S', 'T', None
        # HRF: 'bold', 'spmhrf'

        # Input parameters
        self.method_time = method_time
        self.method_space = method_space
        self.hrf = hrf
        self.Lambda = Lambda
        self.cost_save = cost_save
        self.n_jobs = -2
        self.n_iter = 5

        # Empty fields
        self.data = None
        self.atlas = None
        self.deconvolved_ = None


    def _load_data(self, f, a=None, mask=True, ftype='nifti', detrend=True, standardize=True, highpass=0.01,
                   lowpass=None,
                   TR=2):
        """
        Wrapper for loading all kinds of data
        
        :return: data or data + atlas in 2D 
        """

        if ftype is 'nifti':
            if mask is True:
                cmd = load_nifti
            elif mask is False:
                cmd = load_nifti_nomask

            self.data, self.data_masker, self.atlas, self.atlas_masker = cmd(f, a, detrend=detrend,
                                                                             standardize=standardize,
                                                                             highpass=highpass,
                                                                             lowpass=lowpass, TR=TR)
        elif ftype is 'mat':
            self.data, self.atlas = load_matlab_data(f, a)
        elif ftype is 'txt':
            self.data = load_text_data(f)
            self.atlas = None
        else:
            raise ValueError("Data type not supported. Valid options are 'nifti', 'mat' or 'txt'")

        self.n_voxels = self.data.shape[1]
        self.n_tp = self.data.shape[0]

        self._get_hrf_parameters()

    def _get_hrf_parameters(self):
        """
        Prepares a field with HRF parameters

        :return:
        """

        if self.hrf == 'bold':
            a, psi = hrf.bold_parameters()
        elif self.hrf == 'spmhrf':
            a, psi = hrf.spmhrf_parameters()
        else:
            raise ValueError("HRF must be either bold or spmhrf")

        if self.method_time is 'B':
            self.hrfparams = hrf.block_filter(a, psi, self.TR)
            self.t_iter = 500
        elif self.method_time is 'S':
            self.hrfparams = hrf.spike_filter(a, psi, self.TR)
            self.t_iter = 200
        elif self.method_time is 'W':
            self.hrfparams = hrf.block_filter(a, psi, self.TR)
            self.t_iter = 1
        else:
            raise ValueError('Method_time has to be B, S or W')

    def _temporal(self, d):
        """
        Temporal regularization.
        """


        assert d is not None, "Cannot run anything without loaded data!"


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
        elif self.method_time is 'W':
            self.deconvolved_ = wiener(d, self.hrfparams[0], self.Lambda, self.n_voxels, self.n_tp)
        else:
            print("Wrong temporal deconvolution method; must be B, S or W")

    def _spatial(self, d, a):
        """
        Spatial regularization.
        """

        assert a is not None, "Cannot run spatial regularization without the atlas!"

        if self.method_space is 'T':
            self.deconvolved_ = tikhonov(d, a, self.data_masker, iter=self.s_iter)
        else:
            print("This spatial regularization method is not yet implemented")

    def _deconvolve(self):
        """
        Main control function for deconvolution

        :return:
        """

        if self.method_space is None:
            print("Temporal regularization...")
            self.t_iter *= 5
            t0 = time.time()
            self._temporal(self.data)
            print("Done in %d seconds!" % (time.time() - t0))
        elif self.method_space is 'S':
            print("Structured sparsity spatial regularization not yet implemented")
        elif self.method_space is 'T':
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
            raise ValueError("method_space must be S, T or None")


if __name__ == '__main__':
    ta = TotalActivation()
