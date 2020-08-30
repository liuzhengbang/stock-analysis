# coding=gbk
import math
import random

from net.model import TrainingParam
from utils.consts import index_list_analysis
from utils.csv_utils import *
from data_provider.data_constructor import DataException, \
    construct_dataset, construct_temp_csv_data
from net.trainer import train_model, TrainingDataset, ValidationDataset
from utils.stock_utils import get_code_name_list, get_industry_code_list_in_code_set

thresholds = [0.15]
predict_days = [6]
predict_type = "max"
validate_dataset = True


def filter_list(stock_list):
    print("start filtering stock list")
    for stock in stock_list[::-1]:
        print("processing", stock)
        try:
            construct_dataset(stock, index_list_analysis,
                              predict_days=predict_days, thresholds=thresholds, predict_type=predict_type,
                              return_data=True)
        except DataException:
            print("remove", stock, "from list")
            stock_list.remove(stock)
            continue
    print("total stock num:", len(stock_list))


def split_dataset(dataset, frac=0.8):
    dataset_1 = []
    dataset_2 = []
    for index, row in dataset.iterrows():
        if random.randint(1, 100) <= 100 * frac:
            # print(row)
            dataset_1.append(row)
        else:
            dataset_2.append(row)
    dataset_1 = pd.DataFrame(dataset_1, columns=dataset.columns)
    dataset_2 = pd.DataFrame(dataset_2, columns=dataset.columns)
    dataset_1 = dataset_1.reset_index(drop=True)
    dataset_2 = dataset_2.reset_index(drop=True)
    return dataset_1, dataset_2


def train():
    stock_list = get_industry_code_list_in_code_set(["通信", "电子"], "hs300")

    test_list = ["sz.002456"]
    # test_list = ["sh.600000"]
    stock_list.remove(test_list[0])
    print("total", len(stock_list), "stocks:")
    print(get_code_name_list(stock_list))

    x_test, y_test = construct_dataset(test_list[0], index_list_analysis,
                                       predict_days=predict_days, thresholds=thresholds, predict_type=predict_type,
                                       return_data=True)
    construct_temp_csv_data(stock_list, index_list_analysis,
                            predict_days=predict_days, thresholds=thresholds, predict_type=predict_type)

    pos_data = load_temp_positive_data()
    neg_data = load_temp_negative_data()

    print("total pos sample:", len(pos_data), "neg sample:", len(neg_data))

    if validate_dataset:
        train_pos_data, val_pos_data = split_dataset(pos_data)
        train_neg_data, val_neg_data = split_dataset(neg_data)

        train_dataset = TrainingDataset(train_pos_data, train_neg_data)
        val_dataset = ValidationDataset(val_pos_data, val_neg_data)
    else:
        train_dataset = TrainingDataset(pos_data, neg_data)
        val_dataset = None

    param = TrainingParam(["通信", "电子"], stock_list, test_list, index_list_analysis,
                          predict_days=predict_days, thresholds=thresholds, predict_type=predict_type)
    train_model(train_dataset, val_dataset, x_test, y_test, param,
                prev_model=None,
                num_iterations=8000, learning_rate=0.00001, weight=1, print_cost=True)


train()
