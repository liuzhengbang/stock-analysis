# coding=gbk
import pandas
import torch

from data_provider.data_constructor import construct_dataset, DataException
from net.model import load
from net.trainer import validate, predict
from utils.consts import index_list_analysis
from utils.csv_utils import get_stock_code_list_by_industry
from utils.stock_utils import get_industry_code_list_in_code_set, get_code_name

predict_days = [6]
thresholds = [0.01]
predict_type = "average"


def _validate_model_with_stock_list(model_name, stock_list, index_list_analysis_in,
                                    predict_days_in, thresholds_in, predict_type_in):
    model, _, _, _, _ = load(model_name)
    for stock in stock_list:
        try:
            x_test, y_test = construct_dataset(stock, index_list_analysis_in,
                                               predict_days=predict_days_in, thresholds=thresholds_in,
                                               predict_type=predict_type_in,
                                               return_data=True)
        except DataException:
            continue

        with torch.no_grad():
            code_name = get_code_name(stock)
            accuracy, precision, recall = validate(model, x_test, y_test)
        print("stock {} {}: total sample {}, positive sample {}, accuracy {}%, precision {}%, recall {}%"
              .format(stock, code_name, len(x_test), sum(y_test).item(), accuracy, precision, recall))


def validate_model():
    all_stock_list = get_industry_code_list_in_code_set(["通信", "电子"], "hs300")
    # all_stock_list = get_stock_code_list_by_industry(["通信", "电子"])
    print(all_stock_list)
    _validate_model_with_stock_list("2020-08-30-22-28-43-91.57-33.98-2.15-model_ele_pos_6_max_0.15",
                                    all_stock_list, index_list_analysis,
                                    predict_days_in=predict_days,
                                    thresholds_in=thresholds,
                                    predict_type_in=predict_type)


def save_predict_result(code, model):
    x_test, y_test = construct_dataset(code, index_list_analysis,
                                       predict_days=predict_days, thresholds=thresholds,
                                       predict_type=predict_type,
                                       save_data_to_csv=True,
                                       return_data=True)
    model, _, _, _, _ = load(model)
    y_prediction = predict(model, x_test)
    y = y_prediction.cpu().numpy()
    y = pandas.DataFrame(y)
    y.to_csv("temp/" + code + "_result.csv", header=False, index=True)


# save_predict_result("sz.002475", "2020-08-29-20-29-49-96.32-33.13-0.72-model_ele_pos_6_max_0.2.pt")
validate_model()
