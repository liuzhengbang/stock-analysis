# coding=gbk
import torch

from data_provider.data_constructor import construct_predict_data
from train.trainer import predict_with_prob
from utils.csv_utils import get_stock_code_list_by_industry
from data_provider.data_constructor import convert_to_tensor, construct_dataset, DataException
from train.net import NeuralNetwork as Net

device = torch.device('cuda:0')
model_name = "2020-08-18-05-32-56-84.23-13.22-2.26-model-bank.pt"
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

predict_stock_list = get_stock_code_list_by_industry(["银行"])
model = Net(94).to(device=device)
model.load_state_dict(torch.load("model_data/" + model_name))
model.to(device)

for stock in predict_stock_list:
    try:
        data_x, recent_date = construct_predict_data(stock, index_list_analysis)
    except DataException:
        continue
    ret, prob = predict_with_prob(model, data_x)
    print(stock, ":", ret, "with prob", prob, "on", recent_date)
