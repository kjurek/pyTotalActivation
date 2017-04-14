import numpy as np

from TestBase import TestBase
from TotalActivation.filters.filter_boundary import filter_boundary_normal, filter_boundary_transpose
import unittest

class TestFilterBoundary(TestBase):
    def test_FilterBoundaryNormal(self):
        input = np.array([[1, 5, 2, 4, 3, 3, 1, 5, 2, 4, 3, 3], [5, 10, 3, 3, 4, 5, 7, 8, 10, 15, 12, 11]]).T
        hrfparam = dict(den=np.array([np.array([1.00000000e+00, -6.80326819e-06]), np.array([1.])]),
                        num=np.array([1., -2.63916543, 2.67845911, -1.27148905, 0.24074467, -0.00854931]))
        expected_output = np.array([[1., 5.],
                                    [2.36084138, -3.19579312],
                                    [-8.51735196, -9.99938044],
                                    [10.84241772, 15.5095816],
                                    [-8.31637028, -7.39318055],
                                    [4.44847954, 1.02889834],
                                    [-3.52930207, 1.34029009],
                                    [7.52760082, -1.47122341],
                                    [-11.64374718, 2.21576589],
                                    [11.53898253, 2.3051365],
                                    [-8.34201347, -9.33232082],
                                    [4.44847937, 8.65805982]])
        np.testing.assert_allclose(filter_boundary_normal(hrfparam, input), expected_output)

    def test_FilterBoundaryTranspose(TestBase):
        input = np.array([[1, 5, 2, 4, 3, 3, 1, 5, 2, 4, 3, 3], [5, 10, 3, 3, 4, 5, 7, 8, 10, 15, 12, 11]]).T
        hrfparam = dict(den=np.array([np.array([1.00000000e+00, -6.80326819e-06]), np.array([1.])]),
                        num=np.array([1., -2.63916543, 2.67845911, -1.27148905, 0.24074467, -0.00854931]))
        expected_output = np.array([[-11.22822912, -16.25046992],
                                    [7.33469501, 6.17581024],
                                    [-4.13772595, 1.05570482],
                                    [4.0329613, -1.22431391],
                                    [-8.14910766, 1.66067282],
                                    [12.14740641, -1.25282175],
                                    [-11.22822906, -3.60611193],
                                    [7.34324297, 9.17547747],
                                    [-4.33572349, -11.43226452],
                                    [4.11784761, 12.79294927],
                                    [-4.91747587, -17.03074486],
                                    [3., 11.]])
        np.testing.assert_allclose(filter_boundary_transpose(hrfparam, input), expected_output)

if __name__ == "__main__":
    unittest.main()
