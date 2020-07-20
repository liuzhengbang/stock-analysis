import pandas

from csv_utils import read_individual_csv, read_index_csv
from data_pool.analysis_data import construct_y, construct_x, construct_dataset_with_index
from train.trainer import train_model
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
prepare_data(["sh.600000", "sz.002120", "sz.300142", "sz.300059"],
             index_list_query)

x_train, y_train = construct_dataset_with_index("sz.002120", index_list_analysis)
x_test, y_test = construct_dataset_with_index("sz.300142", index_list_analysis)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


train_model(x_train, y_train, x_test, y_test, num_iterations=300000, learning_rate=0.00001, print_cost=True)

