import pandas
import torch

from csv_utils import read_individual_csv, read_index_csv, get_all_stocks_code
from data_pool.analysis_data import construct_y, construct_x, construct_dataset_with_index, DataException
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
            construct_dataset_with_index(stock, index_list_analysis, filtering_only=True)
        except DataException:
            print("remove", stock, "from list")
            stock_list.remove(stock)
            continue


all_stock_list = get_all_stocks_code()
prepare_data(all_stock_list, index_list_query)
# all_stock_list = ["sz.002120", "sh.600669", "sh.600670"]
filter_list(all_stock_list)


class TrainingDataset(Dataset):
    def __init__(self, stock_list):
        self.stock_list = stock_list

    def __getitem__(self, index):
        # print("dataset get stock", self.stock_list[index])
        x, y = construct_dataset_with_index(self.stock_list[index], index_list_analysis)
        return x, y

    def __len__(self):
        # print("dataset length", len(self.stock_list))
        return len(self.stock_list)


# x_train, y_train = construct_dataset_with_index(all_stock_list[0], index_list_analysis)
x_test, y_test = construct_dataset_with_index("sz.002120", index_list_analysis)
# for code in all_stock_list[1:]:
#
#     try:
#         x_temp, y_temp = construct_dataset_with_index(code, index_list_analysis)
#     except DataException as e:
#         print(e.value, "is dismissed")
#     else:
#         print("processing", code)
#         x_train = torch.cat([x_train, x_temp], dim=0)
#         y_train = torch.cat([y_train, y_temp], dim=0)


# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
dataset = TrainingDataset(all_stock_list)
loader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)

train_model(loader, x_test, y_test, num_iterations=5000, learning_rate=0.00001, print_cost=True)

