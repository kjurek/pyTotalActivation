import numpy as np

from TestBase import TestBase
from TotalActivation.filters import hrf


class MatlabComparisonTestHrfFilters(TestBase):
    def setUp(self):
        self.t_r = 1

    def test_spike_filter_bold(self):
        a, psi = hrf.bold_parameters()
        f_analyze, f_recons, max_eig = hrf.spike_filter(a, psi, self.t_r)

        np.testing.assert_allclose(f_analyze['num'],
                                   [1, -1.639165427154841, 1.039293687162910, -0.232195359716452, 0.008549309479686])
        np.testing.assert_allclose(f_analyze['den'][0], [1, -6.80326819034631e-06])
        np.testing.assert_allclose(f_analyze['den'][1], 1)

        np.testing.assert_allclose(f_recons['num'],
                                   [1, -1.639165427154841, 1.039293687162910, -0.232195359716452, 0.008549309479686])
        np.testing.assert_allclose(f_recons['den'][0], [1, -6.80326819034631e-06])
        np.testing.assert_allclose(f_recons['den'][1], 1)

        np.testing.assert_almost_equal(max_eig, 15.359839612498687)

    def test_spike_filter_spmhrf(self):
        a, psi = hrf.spmhrf_parameters()
        f_analyze, f_recons, max_eig = hrf.spike_filter(a, psi, self.t_r)

        np.testing.assert_allclose(f_analyze['num'],
                                   [1, -2.743302542144524, 2.859320053352163, -1.348960571338628, 0.244289813077911])
        np.testing.assert_allclose(f_analyze['den'][0], [1, -0.874939970605710])
        np.testing.assert_allclose(f_analyze['den'][1], 1)

        np.testing.assert_allclose(f_recons['num'],
                                   [1, -2.743302542144524, 2.859320053352163, -1.348960571338628, 0.244289813077911])
        np.testing.assert_allclose(f_recons['den'][0], [1, -0.874939970605710])
        np.testing.assert_allclose(f_recons['den'][1], 1)

        np.testing.assert_almost_equal(max_eig, 19.107889040927440)

    def test_block_filter_bold(self):
        a, psi = hrf.bold_parameters()
        f_analyze, f_recons, max_eig = hrf.block_filter(a, psi, self.t_r)

        np.testing.assert_allclose(f_analyze['num'],
                                   [1, -2.639165427154841, 2.678459114317751, -1.271489046879362, 0.240744669196138,
                                    -0.008549309479686])
        np.testing.assert_allclose(f_analyze['den'][0], [1, -6.80326819034631e-06])
        np.testing.assert_allclose(f_analyze['den'][1], 1)

        np.testing.assert_allclose(f_recons['num'],
                                   [1, -1.639165427154841, 1.039293687162910, -0.232195359716452, 0.008549309479686])
        np.testing.assert_allclose(f_recons['den'][0], [1, -6.80326819034631e-06])
        np.testing.assert_allclose(f_recons['den'][1], 1)

        np.testing.assert_almost_equal(max_eig, 61.439213877334566)

    def test_block_filter_spmhrf(self):
        a, psi = hrf.spmhrf_parameters()
        f_analyze, f_recons, max_eig = hrf.block_filter(a, psi, self.t_r)

        np.testing.assert_allclose(f_analyze['num'],
                                   [1, -3.743302542144524, 5.602622595496687, -4.208280624690792, 1.593250384416538,
                                    -0.244289813077911])
        np.testing.assert_allclose(f_analyze['den'][0], [1, -0.874939970605710])
        np.testing.assert_allclose(f_analyze['den'][1], 1)

        np.testing.assert_allclose(f_recons['num'],
                                   [1, -2.743302542144524, 2.859320053352163, -1.348960571338628, 0.244289813077911])
        np.testing.assert_allclose(f_recons['den'][0], [1, -0.874939970605710])
        np.testing.assert_allclose(f_recons['den'][1], 1)

        np.testing.assert_almost_equal(max_eig, 76.431376312980860)
