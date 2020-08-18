import unittest

from data_provider.data_constructor import construct_dataset


class TestConstructDataset(unittest.TestCase):
    def test_construct_basic(self):
        construct_dataset("sh.600000", None, return_data=True)