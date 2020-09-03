# coding=gbk
from net.model import TrainingParam
from utils.consts import index_list_analysis
from data_provider.data_constructor import construct_dataset, construct_temp_csv_data, load_dataset
from net.trainer import train_model, TrainingDataset, ValidationDataset
from utils.stock_utils import get_code_name_list, get_stock_code_list_of_industry_contained_in_selected_set

thresholds = [0.12, 0.05]
predict_days = [15, 15]
predict_type = ["max", "average"]


def train():
    industry_list = ["银行"]
    select_list = []
    stock_list = get_stock_code_list_of_industry_contained_in_selected_set(industry_list, select_list)

    test_list = ["sh.600000"]
    stock_list.remove(test_list[0])
    print("total", len(stock_list), "stocks:", get_code_name_list(stock_list))

    x_test, y_test = construct_dataset(test_list[0], index_list_analysis,
                                       predict_days=predict_days, thresholds=thresholds, predict_type=predict_type,
                                       return_data=True)
    construct_temp_csv_data(stock_list, index_list_analysis,
                            predict_days=predict_days, thresholds=thresholds, predict_type=predict_type, val_days=30)

    train_pos_data, train_neg_data, val_pos_data, val_neg_data = load_dataset()
    train_dataset = TrainingDataset(train_pos_data, train_neg_data)
    val_dataset = ValidationDataset(val_pos_data, val_neg_data)

    param = TrainingParam(industry_list, select_list, stock_list, test_list, index_list_analysis,
                          predict_days=predict_days, thresholds=thresholds, predict_type=predict_type)
    train_model(train_dataset, val_dataset, x_test, y_test, param, batch_size=10000,
                prev_model=None,
                num_iterations=10000, learning_rate=0.00001, weight=1, print_cost=True)


train()
