# coding=gbk
import torch

from data_provider.data_constructor import construct_predict_data
from net.model import load, get_prediction_from_param
from net.trainer import predict_with_prob
from utils.consts import index_list_analysis
from utils.csv_utils import get_stock_code_list_by_industry
from data_provider.data_constructor import convert_to_tensor, construct_dataset, DataException
from utils.stock_utils import get_code_name, get_stock_code_list_of_industry_contained_in_selected_set

device = torch.device('cuda:0')

industry_list = ["通信", "电子"]
select_list = ["hs300"]
predict_stock_list = get_stock_code_list_of_industry_contained_in_selected_set(industry_list, select_list)
model, _, _, _, param = load("2020-09-01-03-40-18-86.92-43.86-2.49-model_ele_6_max_0.15")


def predict_stocks(model_loaded, stock_list):
    print(get_prediction_from_param(param))
    for stock in stock_list:
        try:
            data_x, recent_date = construct_predict_data(stock, index_list_analysis)
        except DataException:
            continue
        ret, prob = predict_with_prob(model_loaded, data_x)
        code_name = get_code_name(stock)
        if ret == 1.0:
            print(stock, code_name, ":", ret, "with prob", prob, "on", recent_date)
        # else:
        #     print(stock, code_name, ":", ret, "with prob", prob, "on", recent_date)


predict_stocks(model, predict_stock_list)
