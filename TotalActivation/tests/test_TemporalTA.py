from os.path import join

import numpy as np
import os
import scipy.io as sio

from TestBase import TestBase
from TestBase import traverse_matlab_files
from TotalActivation.process.temporal import temporal_TA
from TotalActivation import TotalActivation

    def test_TemporalTA(self):

        test_ta = TotalActivation()
        test_ta.load_text_data(os.path.join(self.data_path, 'TemporalTA/input_matlab_normalized.csv'),)
        test_ta.data = test_ta.data[:,0:10]
        test_ta.n_voxels = 10
        test_ta.config['TR'] = 1
        test_ta._get_hrf_parameters()
        test_ta.t_iter = 2500
        test_ta._temporal()

        expected_output = np.genfromtxt(os.path.join(self.data_path, 'TemporalTA/phantom_block_nospatial.csv'),
                                        delimiter=',')[:,0:10]

        np.testing.assert_almost_equal(test_ta.deconvolved_, expected_output, decimal = 3)