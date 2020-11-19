import unittest

from dataset.data_constructor import construct_lstm_dataset_to_csv
from model.lstm_trainer import _get_data_position
from utils.csv_utils import delete_temp_data


class TestConstructLstmDataset(unittest.TestCase):
    def setUp(self):
        delete_temp_data()

    def test_construct_basic(self):
        construct_lstm_dataset_to_csv("test", ["index_test"], predict_types=["average"], predict_days=[1],
                                      history_length=2,
                                      predict_thresholds=[0.1],
                                      val_date_list=["2020-08-10"])

    def test_get_data_position(self):
        self.assertEqual(_get_data_position(0, length_list=[1, 3, 17], code_list=["1", "2", "3"]), ("1", 0))
        self.assertEqual(_get_data_position(1, length_list=[1, 3, 17], code_list=["1", "2", "3"]), ("2", 0))
        self.assertEqual(_get_data_position(2, length_list=[1, 3, 17], code_list=["1", "2", "3"]), ("2", 1))
        self.assertEqual(_get_data_position(3, length_list=[1, 3, 17], code_list=["1", "2", "3"]), ("2", 2))
        self.assertEqual(_get_data_position(4, length_list=[1, 3, 17], code_list=["1", "2", "3"]), ("3", 0))
        self.assertEqual(_get_data_position(5, length_list=[1, 3, 17], code_list=["1", "2", "3"]), ("3", 1))
        self.assertEqual(_get_data_position(6, length_list=[1, 3, 17], code_list=["1", "2", "3"]), ("3", 2))



if __name__ == '__main__':
    unittest.main()
