# coding=gbk
from model.model_persistence import TrainingParam
from utils.consts import index_list_analysis
from dataset.data_constructor import construct_dataset_instantly, construct_temp_csv_data, load_dataset, \
    generate_val_data
from model.trainer import train_model, TrainingDataset, ValidationDataset, continue_train
from utils.stock_utils import get_code_name_list, stock_code_list_by_industry_in_constituent

predict_thresholds = [0.10, 0.05]
predict_days = [6, 15]
predict_types = ["max", "average"]


def train():
    industry_list = ["银行"]
    constituent_list = []
    stock_list = stock_code_list_by_industry_in_constituent(industry_list, constituent_list)
    print(stock_list)
    test_list = ["sh.600000"]
    stock_list.remove(test_list[0])
    index_code_list = index_list_analysis

    print(len(stock_list), "stocks:", get_code_name_list(stock_list))

    x_test, y_test = construct_dataset_instantly(test_list[0],
                                                 index_code_list=index_code_list,
                                                 predict_days=predict_days,
                                                 predict_thresholds=predict_thresholds,
                                                 predict_types=predict_types)
    val_date_list = generate_val_data(20000, val_recent_days=30, val_frac=0.2)
    construct_temp_csv_data(stock_list,
                            index_code_list=index_code_list,
                            predict_days=predict_days,
                            thresholds=predict_thresholds,
                            predict_type=predict_types,
                            val_date_list=val_date_list)

    train_pos_data, train_neg_data, val_pos_data, val_neg_data = load_dataset()
    train_dataset = TrainingDataset(train_pos_data, train_neg_data)
    val_dataset = ValidationDataset(val_pos_data, val_neg_data)

    param = TrainingParam(industry_list, constituent_list, stock_list, test_list, index_code_list=index_code_list,
                          val_date_list=val_date_list,
                          predict_days=predict_days, predict_thresholds=predict_thresholds, predict_types=predict_types)
    train_model(train_dataset, val_dataset, x_test, y_test, param, batch_size=10000,
                prev_model=None,
                num_iterations=10000, learning_rate=0.00001, weight=1, print_cost=True)



train()
