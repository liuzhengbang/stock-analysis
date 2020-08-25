import os
import unittest

from data_provider.data_constructor import construct_dataset
from utils.csv_utils import *


class TestConstructDataset(unittest.TestCase):
    def setUp(self):
        delete_temp_data()

    def test_construct_basic(self):
        construct_dataset("test", ["index_test"], predict_days=[2], rolling_days=[1, 3], thresholds=[0])
        pos_data = load_temp_positive_data()
        neg_data = load_temp_negative_data()

        self.assertEqual(len(neg_data), 2)
        self.assertEqual(len(pos_data), 0)

        self.assertSequenceEqual(neg_data.columns.tolist(),
                                 ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ',
                                  'open_index_test', 'high_index_test', 'low_index_test',
                                  'close_index_test', 'volume_index_test', 'amount_index_test',
                                  'pctChg_1', 'volume_1', 'pctChg_3', 'volume_3'])

        self.assertSequenceEqual(neg_data['close'].tolist(), [1.35, 1.62])
        self.assertSequenceEqual(neg_data['pctChg_1'].tolist(), [0.08000000000000003, 0.09000000000000001])
        self.assertSequenceEqual(neg_data['open_index_test'].tolist(), [3.3807621, 3.3705878])


