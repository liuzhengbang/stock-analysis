# coding=gbk
import torch

from data_provider.data_constructor import construct_predict_data
from net.model import load
from net.trainer import predict_with_prob
from utils.consts import index_list_analysis
from utils.csv_utils import get_stock_code_list_by_industry
from data_provider.data_constructor import convert_to_tensor, construct_dataset, DataException
from utils.stock_utils import get_code_name, get_industry_code_list_in_code_set

device = torch.device('cuda:0')

predict_stock_list = get_industry_code_list_in_code_set(["通信", "电子"], "hs300")
# predict_stock_list = get_stock_code_list_by_industry(["银行"])
model, _, _, _, _ = load("2020-08-29-20-29-49-96.32-33.13-0.72-model_ele_pos_6_max_0.2")


def predict_stocks(model_loaded, stock_list):
    for stock in stock_list:
        try:
            data_x, recent_date = construct_predict_data(stock, index_list_analysis)
        except DataException:
            continue
        ret, prob = predict_with_prob(model_loaded, data_x)
        code_name = get_code_name(stock)
        print(stock, code_name, ":", ret, "with prob", prob, "on", recent_date)


predict_stocks(model, predict_stock_list)
