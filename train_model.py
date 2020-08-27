# coding=gbk
import math
import torch
from torch.utils.data.dataloader import DataLoader

from net.model import TrainingParam
from utils.csv_utils import *
from data_provider.data_constructor import DataException, \
    construct_dataset, construct_temp_csv_data
from net.trainer import train_model, validate_model, TrainingDataset

index_list_analysis = ["sh.000001",
                       "sz.399106",
                       "sh.000016",
                       "sh.000300",
                       "sh.000905",
                       "sz.399001",
                       "sh.000015",
                       "sh.000011",
                       "sh.000012",
                       ]

thresholds = [0.1]
predict_days = [6]
predict_type = "max"


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


def train():
    stock_list = get_stock_code_list_by_industry(["银行"])
    test_list = ["sh.600000"]
    stock_list.remove("sh.600000")
    construct_temp_csv_data(stock_list, index_list_analysis,
                            predict_days=predict_days, thresholds=thresholds, predict_type=predict_type)
    x_test, y_test = construct_dataset("sh.600000", index_list_analysis,
                                       predict_days=predict_days, thresholds=thresholds, predict_type=predict_type,
                                       return_data=True)

    pos_dataset = load_temp_positive_data()
    neg_dataset = load_temp_negative_data()
    dataset = TrainingDataset(pos_dataset, neg_dataset)
    loader = DataLoader(dataset, batch_size=2000, num_workers=0, shuffle=False)
    param = TrainingParam(["银行"], stock_list, test_list, index_list_analysis,
                          predict_days=predict_days, thresholds=thresholds, predict_type=predict_type,)
    train_model(loader, x_test, y_test, param, prev_model="2020-08-27-23-31-41-94.39-61.96-4.22-model.pt",
                num_iterations=100, learning_rate=0.00001, weight=1, print_cost=True)


train()
# all_stock_list = get_stock_code_list_by_industry(["银行"])
# validate_model("2020-08-27-23-31-41-94.39-61.96-4.22-model.pt",
#                all_stock_list, index_list_analysis,
#                predict_days=predict_days, thresholds=thresholds, predict_type=predict_type)
