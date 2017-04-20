from os.path import join

import numpy as np
import os

from TestBase import TestBase
from TotalActivation import TotalActivation


class MatlabComparisonTestTemporalTA(TestBase):

    def test_TemporalTA_nocostsave(self):
        test_ta = TotalActivation()
        test_ta.load_text_data(os.path.join(self.data_path, 'TemporalTA/input_matlab_normalized.csv'),)
        test_ta.data = test_ta.data[:,0:10]
        test_ta.n_voxels = 10
        test_ta.config['TR'] = 1
        test_ta.config['cost_save'] = False
        test_ta._get_hrf_parameters()
        test_ta.t_iter = 2500
        test_ta._temporal()

        expected_output = np.genfromtxt(os.path.join(self.data_path, 'TemporalTA/phantom_block_nospatial.csv'),
                                        delimiter=',')[:,0:10]

        np.testing.assert_almost_equal(test_ta.deconvolved_, expected_output, decimal = 3)


    def test_TemporalTA_costsave(self):
        test_ta = TotalActivation()
        test_ta.load_text_data(os.path.join(self.data_path, 'TemporalTA/input_matlab_normalized.csv'), )
        test_ta.data = test_ta.data[:, 0:10]
        test_ta.n_voxels = 10
        test_ta.config['TR'] = 1
        test_ta.config['cost_save'] = True
        test_ta._get_hrf_parameters()
        test_ta.t_iter = 2500
        test_ta._temporal()

        expected_output = np.genfromtxt(os.path.join(self.data_path, 'TemporalTA/phantom_block_nospatial.csv'),
                                        delimiter=',')[:, 0:10]

        np.testing.assert_almost_equal(test_ta.deconvolved_, expected_output, decimal=3)