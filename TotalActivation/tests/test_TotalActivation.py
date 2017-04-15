from __future__ import absolute_import, division, print_function
import os.path as op
import unittest

import TestBase

data_path = op.join(sb.__path__[0], 'data')

class TestTotalActivaton(TestBase):
    def test_TotalActivation(selfs):
        # dummy test
        np.testing.assert_allclose(1, 1)

if __name__ == "__main__":
    unittest.main()
