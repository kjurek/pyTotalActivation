import unittest
import scipy.io as sio
import numpy as np
import os
import random
from TestBase import TestBase
from nilearn import signal
import matplotlib.pyplot as plt


class MatlabComparisonTestDetrend(TestBase):

    def test_detrend_normalize_mat(self):
        from detrend import detrend_normalize_mat
        test_data = sio.loadmat(os.path.join(self.data_path, 'detrend', 'MyDetrend_normalize_mat.mat'))
        result = detrend_normalize_mat(test_data['TC'])
        self.assertEquals(test_data['TCN'].shape, result.shape)
        np.testing.assert_allclose(test_data['TCN'], result, rtol=0.01)

    def test_detrend_normalize_nii(self):
        from detrend import detrend_normalize_nii
        test_data = sio.loadmat(os.path.join(self.data_path, 'detrend', 'MyDetrend_normalize_nii.mat'))
        result = detrend_normalize_nii(test_data['TC'])
        self.assertEquals(test_data['TCN'].shape, result.shape)
        np.testing.assert_allclose(test_data['TCN'], result, rtol=0.01)

    def test_detrend_from_nilearn(self):
        test_data = sio.loadmat(os.path.join(self.data_path, 'detrend', 'MyDetrend_normalize_nii.mat'))
        result = signal.clean(signals=np.transpose(test_data['TC']), t_r=1, detrend=False)
        self.assertEquals(test_data['TCN'].shape, result.shape)
        np.testing.assert_allclose(test_data['TCN'], result, rtol=0.01)

        """
        random_signal_index = random.choice(range(result.shape[0]))
        t = np.arange(0, test_data['TC'].shape[1])
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.plot(t, test_data['TCN'][:, random_signal_index], t, result[:, random_signal_index])
        plt.legend(['MATLAB', 'nilearn.signal.clean'])
        plt.show()
        """


if __name__ == '__main__':
    unittest.main()
