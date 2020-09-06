# coding=gbk
import torch

from dataset.data_constructor import construct_predict_data
from model.model_persistence import load, get_prediction_from_param
from model.trainer import predict_with_prob
from utils.consts import index_list_analysis
from utils.csv_utils import get_stock_code_list_by_industry
from dataset.data_constructor import df_to_tensor, construct_dataset_instantly, DataException
from utils.stock_utils import get_code_name, stock_code_list_by_industry_in_constituent

device = torch.device('cuda:0')

model, _, _, _, param = load("20200906-133514-82.4-15.3-model_hs300")


def predict_stocks(model_loaded, industry_list=None, select_set=None):
    print(get_prediction_from_param(param))

    if industry_list is None:
        industry_list = param.get_industry_list()

    if select_set is None:
        select_set = param.get_constituent()

    stock_list = stock_code_list_by_industry_in_constituent(industry_list, select_set)
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


predict_stocks(model)
