import os
import unittest

from data_provider.data_constructor import construct_dataset
from utils.csv_utils import *


class TestConstructDataset(unittest.TestCase):
    def setUp(self):
        delete_temp_data()

    def test_construct_basic(self):
        construct_dataset("test", ["index_test"], predict_days=[1], rolling_days=[1], thresholds=[0])
        pos_data = load_temp_positive_data()
        neg_data = load_temp_negative_data()

        self.assertEqual(len(neg_data), 1)
        self.assertEqual(len(pos_data), 0)

        self.assertSequenceEqual(neg_data.columns.tolist(),
                                 ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ',
                                  'open_index_test', 'high_index_test', 'low_index_test',
                                  'close_index_test', 'volume_index_test', 'amount_index_test',
                                  'pctChg_1', 'volume_1'])

        self.assertSequenceEqual(neg_data['close'].tolist(), [1.15])
        self.assertSequenceEqual(neg_data['pctChg_1'].tolist(), [0.06])
        self.assertSequenceEqual(neg_data['open_index_test'].tolist(), [3.3281754])


    def test_construct_basic_2(self):
        construct_dataset("test", ["index_test"], predict_days=[1], rolling_days=[1], thresholds=[0])
