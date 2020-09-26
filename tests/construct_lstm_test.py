import unittest

from dataset.data_constructor import construct_lstm_dataset_to_csv
from utils.csv_utils import delete_temp_data


class TestConstructLstmDataset(unittest.TestCase):
    def setUp(self):
        delete_temp_data()

    def test_construct_basic(self):
        construct_lstm_dataset_to_csv("test", ["index_test"], predict_types=["average"], predict_days=[1],
                                      history_length=4,
                                      predict_thresholds=[0.1])


if __name__ == '__main__':
    unittest.main()
