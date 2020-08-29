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

predict_stock_list = get_industry_code_list_in_code_set(["ͨ��", "����"], "hs300")
model, _, _, _, _ = load("2020-08-29-23-22-22-76.69-30.6-9.91-model.pt")

for stock in predict_stock_list:
    try:
        data_x, recent_date = construct_predict_data(stock, index_list_analysis)
    except DataException:
        continue
    ret, prob = predict_with_prob(model, data_x)
    code_name = get_code_name(stock)
    print(stock, code_name, ":", ret, "with prob", prob, "on", recent_date)
