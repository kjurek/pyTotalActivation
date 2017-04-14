import unittest
import os


def traverse_matlab_files(path):
    for root, dirs, files in os.walk(path):
        for fname in [f for f in files if f.endswith('.mat')]:
            yield os.path.join(root, fname)


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(__file__), 'test_data')
