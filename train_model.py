# coding=gbk
import math
import torch
from torch.utils.data.dataloader import DataLoader

from utils.csv_utils import *
from data_provider.data_constructor import DataException, \
    construct_dataset, construct_temp_csv_data
from train.trainer import train_model, validate_model
from torch.utils.data.dataset import Dataset
from stock_query.stock import prepare_data


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


def filter_list(stock_list):
    print("start filtering stock list")
    for stock in stock_list[::-1]:
        print("processing", stock)
        try:
            construct_dataset(stock, index_list_analysis, return_data=True)
        except DataException:
            print("remove", stock, "from list")
            stock_list.remove(stock)
            continue
    print("total stock num:", len(stock_list))


# need_refresh_data = False
#
# if need_refresh_data:
#     all_stock_list = get_all_stocks_code_list()
# else:
#     all_stock_list = load_filtered_stock_list()

# all_stock_list = get_stock_code_list_by_industry(["银行"])


# construct_temp_csv_data(all_stock_list, index_list_analysis)
# filter_list(all_stock_list)

# all_stock_list = ["sz.002120", "sh.600600", "sh.600601"]
# all_stock_list = ["test"]


class TrainingDataset(Dataset):
    def __init__(self, positive_data, negative_data):
        self.pos_data = positive_data
        self.neg_data = negative_data
        self.pos_length = len(self.pos_data)
        self.neg_length = len(self.neg_data)
        self.length = max(self.pos_length, self.neg_length) * 2

    def __getitem__(self, ndx):
        index = ndx // 2
        # print("index", index, "in", ndx)
        if ndx % 2 == 0:
            pos_index = index % self.pos_length
            # print("pos index", pos_index)
            return self.pos_data.values[pos_index], torch.tensor([1.0])
        else:
            neg_index = index % self.neg_length
            # print("neg index", neg_index)
            return self.neg_data.values[neg_index], torch.tensor([0.0])

    def __len__(self):
        # print("length", self.length)
        return self.length


def train():
    stock_list = get_stock_code_list_by_industry(["银行"])
    stock_list.remove("sh.600000")
    # construct_temp_csv_data(stock_list, index_list_analysis)
    x_test, y_test = construct_dataset("sh.600000", index_list_analysis, return_data=True)
    # # # x_test, y_test = construct_dataset("test", index_list_analysis)
    # # print("all stocks being trained", all_stock_list)
    pos_dataset = load_temp_positive_data()
    neg_dataset = load_temp_negative_data()
    dataset = TrainingDataset(pos_dataset, neg_dataset)
    loader = DataLoader(dataset, batch_size=2000, num_workers=0, shuffle=False)
    train_model(loader, x_test, y_test, prev_model=None,
                num_iterations=1000, learning_rate=0.00001, weight=1, print_cost=True)


train()
# all_stock_list = get_stock_code_list_by_industry(["银行"])
# validate_model("2020-08-25-22-27-56-81.08-5.71-1.27-model.pt",
#                all_stock_list, index_list_analysis)
