import numpy as np
import os
import scipy.io as sio
import unittest
from nilearn import signal

from TestBase import TestBase


class MatlabComparisonTestDetrend(TestBase):
    def test_detrend_normalize_mat(self):
        from TotalActivation.preprocess.detrend import detrend_normalize_mat
        test_data = sio.loadmat(os.path.join(self.data_path, 'detrend', 'MyDetrend_normalize_mat.mat'))
        result = detrend_normalize_mat(test_data['TC'])
        self.assertEquals(test_data['TCN'].shape, result.shape)
        np.testing.assert_allclose(test_data['TCN'], result, rtol=0.01)

    def test_detrend_normalize_nii(self):
        from TotalActivation.preprocess.detrend import detrend_normalize_nii
        test_data = sio.loadmat(os.path.join(self.data_path, 'detrend', 'MyDetrend_normalize_nii.mat'))
        result = detrend_normalize_nii(test_data['TC'])
        self.assertEquals(test_data['TCN'].shape, result.shape)
        np.testing.assert_allclose(test_data['TCN'], result, rtol=0.01)

    def test_detrend_from_nilearn(self):
        test_data = sio.loadmat(os.path.join(self.data_path, 'detrend', 'MyDetrend_normalize_nii.mat'))
        result = signal.clean(signals=np.transpose(test_data['TC']), t_r=1, detrend=False)
        self.assertEquals(test_data['TCN'].shape, result.shape)
        np.testing.assert_allclose(test_data['TCN'], result, rtol=0.01)


if __name__ == '__main__':
    unittest.main()
