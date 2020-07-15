from data_pool.analysis_data import construct_train_y, construct_train_x
from train.trainer import train_model
from stock_query.stock import prepare_data

index_list = ["sh.000001",
              "sz.399106",
              "sh.000016",
              "sh.000300",
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
prepare_data(["sh.600000", "sz.002120", "sz.300142", "sz.300059"],
             index_list)

# x_train = construct_train_x("sz.002120")
# y_train = construct_train_y("sz.002120")
#
# x_test = construct_train_x("sz.300142")
# y_test = construct_train_y("sz.300142")
#
# train_model(x_train, y_train, x_test, y_test, num_iterations=20000, learning_rate=0.00001, print_cost=True)
