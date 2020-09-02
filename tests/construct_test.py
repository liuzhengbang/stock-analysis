import os
import unittest

from data_provider.data_constructor import construct_dataset
from utils.csv_utils import *


class TestConstructDataset(unittest.TestCase):
    def setUp(self):
        delete_temp_data()

    def test_construct_basic(self):
        construct_dataset("test", ["index_test"], predict_type="average", predict_days=[1], rolling_days=[2, 4],
                          append_index=True,
                          val_days=0,
                          thresholds=[0.1])
        pos_data = load_temp_data(POSITIVE_CSV)
        neg_data = load_temp_data(NEGATIVE_CSV)
        #          date  result  close  chg_1
        # 0  2020-08-07     0.0    2.0    0.0
        # 1  2020-08-10     1.0    2.0    0.1
        # 2  2020-08-11     0.0    2.2    NaN

        self.assertEqual(len(neg_data), 1)
        self.assertEqual(len(pos_data), 1)

        self.assertSequenceEqual(neg_data.columns.tolist(),
                                 ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ',
                                  'open_index_test', 'high_index_test', 'low_index_test',
                                  'close_index_test', 'volume_index_test', 'amount_index_test',
                                  'pctChg_2', 'volume_2', 'ma_2', 'highest_2', 'lowest_2', 'peTTM_2',
                                  'pbMRQ_2', 'pctChg_4', 'volume_4', 'ma_4', 'highest_4', 'lowest_4', 'peTTM_4',
                                  'pbMRQ_4'])

        self.assertSequenceEqual(neg_data['open'].tolist(), [1.4])
        self.assertSequenceEqual(neg_data['low'].tolist(), [1.3])
        self.assertSequenceEqual(neg_data['volume'].tolist(), [5.5])
        self.assertAlmostEqual(neg_data['pctChg_2'].tolist()[0], (0.08 + 0.09))
        self.assertAlmostEqual(neg_data['volume_2'].tolist()[0], (4.5 + 5.5) / 2)
        self.assertAlmostEqual(neg_data['pctChg_4'].tolist()[0], (0.06 + 0.07 + 0.08 + 0.09))
        self.assertAlmostEqual(neg_data['volume_4'].tolist()[0], (2.5 + 3.5 + 4.5 + 5.5) / 4)
        self.assertAlmostEqual(neg_data['open_index_test'].tolist()[0], 3.3705878)

        self.assertSequenceEqual(pos_data['open'].tolist(), [1.5])
        self.assertSequenceEqual(pos_data['low'].tolist(), [1.4])
        self.assertSequenceEqual(pos_data['volume'].tolist(), [6.5])
        self.assertAlmostEqual(pos_data['pctChg_2'].tolist()[0], (0.09 + 0.10))
        self.assertAlmostEqual(pos_data['volume_2'].tolist()[0], (5.5 + 6.5) / 2)
        self.assertAlmostEqual(pos_data['pctChg_4'].tolist()[0], (0.07 + 0.08 + 0.09 + 0.10))
        self.assertAlmostEqual(pos_data['volume_4'].tolist()[0], (3.5 + 4.5 + 5.5 + 6.5) / 4)
        self.assertAlmostEqual(pos_data['open_index_test'].tolist()[0], 3.3415276)

    def test_construct_predict_average_pos(self):
        construct_dataset("test_1", ["index_test"], predict_type="average", predict_days=[1, 3], rolling_days=[2],
                          append_index=True,
                          val_days=0,
                          thresholds=[0.07, 0.01])
        pos_data = load_temp_data(POSITIVE_CSV)
        neg_data = load_temp_data(NEGATIVE_CSV)

        #          date  result  close     chg_1     chg_3
        # 0  2020-08-05     1.0  0.100  0.100000  0.076667
        # 1  2020-08-06     0.0  0.110  0.009091 -0.021212
        # 2  2020-08-07     0.0  0.111 -0.081081 -0.018018
        # 3  2020-08-10     1.0  0.102  0.078431  0.094771
        # 4  2020-08-11     0.0  0.110  0.045455  0.012121
        # 5  2020-08-12     0.0  0.115 -0.043478       NaN
        # 6  2020-08-13     0.0  0.110 -0.009091       NaN
        # 7  2020-08-14     0.0  0.109       NaN       NaN

        self.assertEqual(len(neg_data), 3)
        self.assertEqual(len(pos_data), 2)

        self.assertSequenceEqual(neg_data.columns.tolist(),
                                 ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ',
                                  'open_index_test', 'high_index_test', 'low_index_test',
                                  'close_index_test', 'volume_index_test', 'amount_index_test',
                                  'pctChg_2', 'volume_2', 'ma_2', 'highest_2', 'lowest_2', 'peTTM_2',
                                  'pbMRQ_2'])

    def test_construct_predict_average_neg(self):
        construct_dataset("test_1", ["index_test"], predict_type="average", predict_days=[1, 3], rolling_days=[2],
                          append_index=True,
                          val_days=0,
                          thresholds=[-0.08, -0.01])
        pos_data = load_temp_data(POSITIVE_CSV)
        neg_data = load_temp_data(NEGATIVE_CSV)

        #          date  result  close     chg_1     chg_3
        # 0  2020-08-05     0.0  0.100  0.100000  0.076667
        # 1  2020-08-06     0.0  0.110  0.009091 -0.021212
        # 2  2020-08-07     1.0  0.111 -0.081081 -0.018018
        # 3  2020-08-10     0.0  0.102  0.078431  0.094771
        # 4  2020-08-11     0.0  0.110  0.045455  0.012121
        # 5  2020-08-12     0.0  0.115 -0.043478       NaN
        # 6  2020-08-13     0.0  0.110 -0.009091       NaN
        # 7  2020-08-14     0.0  0.109       NaN       NaN

        self.assertEqual(len(neg_data), 4)
        self.assertEqual(len(pos_data), 1)
        self.assertSequenceEqual(neg_data.columns.tolist(),
                                 ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ',
                                  'open_index_test', 'high_index_test', 'low_index_test',
                                  'close_index_test', 'volume_index_test', 'amount_index_test',
                                  'pctChg_2', 'volume_2', 'ma_2', 'highest_2', 'lowest_2', 'peTTM_2', 'pbMRQ_2'])

    def test_construct_predict_average_equal(self):
        construct_dataset("test_1", ["index_test"], predict_type="average",
                          append_index=True,
                          predict_days=[1, 3], rolling_days=[2],
                          val_days=0, thresholds=[0, 0])
        pos_data = load_temp_data(POSITIVE_CSV)
        neg_data = load_temp_data(NEGATIVE_CSV)

        #          date  result  close     chg_1     chg_3
        # 0  2020-08-05     1.0  0.100  0.100000  0.076667
        # 1  2020-08-06     0.0  0.110  0.009091 -0.021212
        # 2  2020-08-07     0.0  0.111 -0.081081 -0.018018
        # 3  2020-08-10     1.0  0.102  0.078431  0.094771
        # 4  2020-08-11     1.0  0.110  0.045455  0.012121
        # 5  2020-08-12     0.0  0.115 -0.043478       NaN
        # 6  2020-08-13     0.0  0.110 -0.009091       NaN
        # 7  2020-08-14     0.0  0.109       NaN       NaN

        self.assertEqual(len(neg_data), 2)
        self.assertEqual(len(pos_data), 3)

        self.assertSequenceEqual(neg_data.columns.tolist(),
                                 ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ',
                                  'open_index_test', 'high_index_test', 'low_index_test',
                                  'close_index_test', 'volume_index_test', 'amount_index_test',
                                  'pctChg_2', 'volume_2', 'ma_2', 'highest_2', 'lowest_2', 'peTTM_2', 'pbMRQ_2'])

    def test_construct_predict_max_pos(self):
        construct_dataset("test_2", ["index_test"], predict_type="max",
                          append_index=True,
                          predict_days=[1, 3], rolling_days=[2], val_days=0, thresholds=[0.07, 0.01])
        pos_data = load_temp_data(POSITIVE_CSV)
        neg_data = load_temp_data(NEGATIVE_CSV)

        #          date  result  close   high     chg_1     chg_3
        # 0  2020-08-05     1.0  0.100  0.110  0.130000  0.130000
        # 1  2020-08-06     0.0  0.110  0.113  0.009091  0.009091
        # 2  2020-08-07     0.0  0.111  0.111 -0.072072  0.009009
        # 3  2020-08-10     1.0  0.102  0.103  0.078431  0.098039
        # 4  2020-08-11     0.0  0.110  0.110  0.018182  0.018182
        # 5  2020-08-12     0.0  0.115  0.112 -0.034783       NaN
        # 6  2020-08-13     0.0  0.110  0.111  0.000000       NaN
        # 7  2020-08-14     0.0  0.109  0.110       NaN       NaN

        self.assertEqual(len(neg_data), 3)
        self.assertEqual(len(pos_data), 2)

        self.assertSequenceEqual(neg_data.columns.tolist(),
                                 ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ',
                                  'open_index_test', 'high_index_test', 'low_index_test',
                                  'close_index_test', 'volume_index_test', 'amount_index_test',
                                  'pctChg_2', 'volume_2', 'ma_2', 'highest_2', 'lowest_2', 'peTTM_2', 'pbMRQ_2'])

    def test_construct_predict_max_neg(self):
        construct_dataset("test_2", ["index_test"], predict_type="max",
                          append_index=True,
                          predict_days=[1, 3], rolling_days=[2], val_days=0,
                          thresholds=[-0.04, -0.02])
        pos_data = load_temp_data(POSITIVE_CSV)
        neg_data = load_temp_data(NEGATIVE_CSV)

        #          date  result  close    low     chg_1     chg_3
        # 0  2020-08-05     0.0  0.100  0.090  0.000000  0.000000
        # 1  2020-08-06     0.0  0.110  0.100  0.000000 -0.081818
        # 2  2020-08-07     1.0  0.111  0.110 -0.090090 -0.090090
        # 3  2020-08-10     0.0  0.102  0.101  0.078431  0.009804
        # 4  2020-08-11     1.0  0.110  0.110 -0.045455 -0.063636
        # 5  2020-08-12     0.0  0.115  0.105 -0.104348       NaN
        # 6  2020-08-13     0.0  0.110  0.103 -0.027273       NaN
        # 7  2020-08-14     0.0  0.109  0.107       NaN       NaN

        self.assertEqual(len(neg_data), 3)
        self.assertEqual(len(pos_data), 2)

        self.assertSequenceEqual(neg_data.columns.tolist(),
                                 ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ',
                                  'open_index_test', 'high_index_test', 'low_index_test',
                                  'close_index_test', 'volume_index_test', 'amount_index_test',
                                  'pctChg_2', 'volume_2', 'ma_2', 'highest_2', 'lowest_2', 'peTTM_2', 'pbMRQ_2'])

    def test_construct_predict_max_equal(self):
        construct_dataset("test_2", ["index_test"], predict_type="max",
                          append_index=True,
                          predict_days=[1, 3], rolling_days=[2], val_days=0,
                          thresholds=[0, 0])
        pos_data = load_temp_data(POSITIVE_CSV)
        neg_data = load_temp_data(NEGATIVE_CSV)

        #          date  result  close    low     chg_1     chg_3
        # 0  2020-08-05     1.0  0.100  0.090  0.130000  0.130000
        # 1  2020-08-06     1.0  0.110  0.100  0.009091  0.009091
        # 2  2020-08-07     0.0  0.111  0.110 -0.072072  0.009009
        # 3  2020-08-10     1.0  0.102  0.101  0.078431  0.098039
        # 4  2020-08-11     1.0  0.110  0.110  0.018182  0.018182
        # 5  2020-08-12     0.0  0.115  0.105 -0.034783       NaN
        # 6  2020-08-13     0.0  0.110  0.103  0.000000       NaN
        # 7  2020-08-14     0.0  0.109  0.107       NaN       NaN

        self.assertEqual(len(neg_data), 1)
        self.assertEqual(len(pos_data), 4)

        self.assertSequenceEqual(neg_data.columns.tolist(),
                                 ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ',
                                  'open_index_test', 'high_index_test', 'low_index_test',
                                  'close_index_test', 'volume_index_test', 'amount_index_test',
                                  'pctChg_2', 'volume_2', 'ma_2', 'highest_2', 'lowest_2', 'peTTM_2', 'pbMRQ_2'])
