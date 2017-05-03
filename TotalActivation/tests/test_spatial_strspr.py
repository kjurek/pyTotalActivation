from os.path import join

import numpy as np
import scipy.io as sio

from TestBase import TestBase, traverse_matlab_files
from TotalActivation.process.spatial import strspr, strspr_mat, clip, atlas_filter, atlas_filter_cconj, atlas_filter_conj


class MatlabComparisonTestSpatialStrSpr(TestBase):
    def setUp(self):
        self.data_path = join(self.data_path, 'Spatial')

    def test_clip(self):
        for test_file in traverse_matlab_files(join(self.data_path, 'clip')):
            test_data = sio.loadmat(test_file, struct_as_record=False, squeeze_me=True)
            out = clip(test_data['in'], test_data['atlas'])
            np.testing.assert_allclose(out, test_data['out'])

    def test_atlas_filter(self):
        for test_file in traverse_matlab_files(join(self.data_path, 'atlas_filter', 'regular')):
            test_data = sio.loadmat(test_file, struct_as_record=False, squeeze_me=True)
            out = atlas_filter(test_data['in'], test_data['h'])
            np.testing.assert_allclose(out, test_data['atlas_out'])

    def test_atlas_filter_conj(self):
        for test_file in traverse_matlab_files(join(self.data_path, 'atlas_filter', 'conj')):
            test_data = sio.loadmat(test_file, struct_as_record=False, squeeze_me=True)
            out = atlas_filter_conj(test_data['in'], test_data['h'])
            np.testing.assert_allclose(out, test_data['atlas_out'])

    def test_atlas_filter_cconj(self):
        for test_file in traverse_matlab_files(join(self.data_path, 'atlas_filter', 'cconj')):
            test_data = sio.loadmat(test_file, struct_as_record=False, squeeze_me=True)
            out = atlas_filter_cconj(test_data['in'], test_data['h'])
            np.testing.assert_allclose(out, test_data['atlas_out'])

    def test_strspr_mat(self):
        input = sio.loadmat(join(self.data_path, 'input_mat.mat'), struct_as_record=False, squeeze_me=True)
        expected_output = sio.loadmat(join(self.data_path, 'output_mat.mat'), struct_as_record=False, squeeze_me=True)
        x_out = strspr_mat(input['y'], input['atlas'], input['LambdaSpat'], input['NitSpat'], input['VoxelIdx'] - 1,
                           input['Dimension'])
        np.testing.assert_allclose(x_out, expected_output['x_out'])

    def test_strspr(self):
        input = sio.loadmat(join(self.data_path, 'input_nii.mat'), struct_as_record=False, squeeze_me=True)
        expected_output = sio.loadmat(join(self.data_path, 'output_nii.mat'), struct_as_record=False, squeeze_me=True)
        x_out = strspr(input['y'], input['atlas'], input['LambdaSpat'], input['NitSpat'], input['VoxelIdx'] - 1,
                       input['Dimension'])
        np.testing.assert_allclose(x_out, expected_output['x_out'])


if __name__ == '__main__':
    import unittest

    unittest.main()
