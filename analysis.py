from data_pool.analysis_data import construct_train_y, construct_train_x
from logistic_regression.logistic_regression import train_model
from stock_query.stock import prepare_data

# prepare_data(["sh.600000", "sz.002120", "sz.300142"])

x_train = construct_train_x("sz.002120")
y_train = construct_train_y("sz.002120")

x_test = construct_train_x("sz.300142")
y_test = construct_train_y("sz.300142")

train_model(x_train, y_train, x_test, y_test, num_iterations=2000000, learning_rate=0.001, print_cost=True)
