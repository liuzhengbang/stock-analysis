import math

import torch

from utils.csv_utils import get_all_stocks_code, save_filtered_stock_list, \
    load_filtered_stock_list
from data_provider.data_constructor import DataException, \
    construct_dataset_with_index_and_history, construct_dataset_batch
from train.trainer import train_model
from torch.utils.data.dataset import Dataset

from stock_query.stock import prepare_data

index_list_query = ["sh.000001",
                    "sz.399106",
                    "sh.000016",
                    "sh.000300",
                    "sh.000905",
                    "sz.399001",
                    "sh.000037",
                    "sz.399433",
                    "sh.000952",
                    "sh.000050",
                    "sh.000982",
                    "sh.000029",
                    "sh.000015",
                    "sh.000063",
                    "sh.000011",
                    "sh.000012",
                    ]

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
        # print("processing", stock)
        try:
            construct_dataset_with_index_and_history(stock, index_list_analysis, filtering_only=True)
        except DataException:
            print("remove", stock, "from list")
            stock_list.remove(stock)
            continue
    print("total stock num:", len(stock_list))


need_refresh_data = False
append_mode = True

if need_refresh_data:
    all_stock_list = get_all_stocks_code()
    filter_list(all_stock_list)
    save_filtered_stock_list(all_stock_list)

    prepare_data(all_stock_list, index_list_query, append=append_mode)
else:
    all_stock_list = load_filtered_stock_list()


# all_stock_list = ["sz.002120", "sh.600600", "sh.600601"]
all_stock_list = ["test"]


class TrainingDataset(Dataset):
    def __init__(self, stock_list, num_stock_per_batch=100):
        self.num_stock_per_batch = num_stock_per_batch
        self.stock_list = stock_list

    def __getitem__(self, index):
        start = index * self.num_stock_per_batch
        end = (index + 1) * self.num_stock_per_batch
        if end >= len(self.stock_list):
            end = len(self.stock_list)
        x, y = construct_dataset_batch(self.stock_list[start:end], index_list_analysis)
        return x, y

    def __len__(self):
        return math.ceil(len(self.stock_list)/self.num_stock_per_batch)


x_test, y_test = construct_dataset_with_index_and_history("test", index_list_analysis)


dataset = TrainingDataset(all_stock_list, num_stock_per_batch=30)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

train_model(loader, x_test, y_test, num_iterations=5000, learning_rate=0.000001, print_cost=True)

