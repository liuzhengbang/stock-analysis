# coding=gbk
from dataset.data_constructor import construct_predict_data
from model.model_persistence import load, get_prediction_from_param
from model.trainer import predict_with_prob
from utils.consts import index_list_analysis
from utils.csv_utils import get_stock_code_list_by_industry
from dataset.data_constructor import DataException
from utils.stock_utils import get_code_name, stock_code_list_by_industry_in_constituent


def predict_stock_list():
    model_list = [
        "20200907-050903-87.2-56.9-model@ele_hs300-pos",
        "20200908-190122-85.1-52.1-model@ele_hs300zz500-pos",
        "20200908-224657-72.5-56.5-model@ele_hs300zz500-neg",
        "20200907-202358-98.6-66.7-model@bank-pos",
        "20200909-215434-98.4-71.9-model@bank-pos",
        "20200908-230852-96.6-61.0-model@sz50-pos",
        "20200909-061343-98.0-84.0-model@sz50-pos",
        "20200912-183550-98.6-54.0-model@hs300-pos"
    ]
    for model in model_list:
        predict_stocks(model)
        print("")


def predict_stocks(model_name, industry_list=None, select_set=None):
    model, _, _, _, _, param = load(model_name)
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
        ret, prob = predict_with_prob(model, data_x)
        code_name = get_code_name(stock)
        if ret == 1.0:
            print(stock, code_name, ":", ret, "with prob", prob, "on", recent_date)
        # else:
        #     print(stock, code_name, ":", ret, "with prob", prob, "on", recent_date)


predict_stocks("20200912-180913-98.6-52.2-model")
# predict_stock_list()
