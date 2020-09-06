# coding=gbk
import pandas
import torch

from dataset.data_constructor import construct_dataset_instantly, DataException
from model.model_persistence import load
from model.trainer import validate, predict
from utils.consts import index_list_analysis
from utils.csv_utils import get_stock_code_list_by_industry
from utils.stock_utils import stock_code_list_by_industry_in_constituent, get_code_name


def validate_model():
    predict_thresholds = [0.05]
    predict_days = [15]
    predict_types = ["max"]
    val_days = 30
    model, _, _, _, param = load("20200906-213432-94.1-50.5-model")
    industry_list = param.get_industry_list()
    select_set = param.get_constituent()

    all_stock_list = stock_code_list_by_industry_in_constituent(
        industry_list, select_set)
    print(all_stock_list)
    _validate_model_with_stock_list(model,
                                    stock_list=all_stock_list, index_list_analysis_in=index_list_analysis,
                                    val_days=val_days,
                                    predict_days_in=predict_days,
                                    thresholds_in=predict_thresholds,
                                    predict_type_in=predict_types)


def _validate_model_with_stock_list(model, stock_list, index_list_analysis_in,
                                    val_days,
                                    predict_days_in, thresholds_in, predict_type_in):

    for stock in stock_list:
        try:
            x_test, y_test = construct_dataset_instantly(stock, index_list_analysis_in,
                                                         predict_days=predict_days_in,
                                                         predict_thresholds=thresholds_in,
                                                         val_days=val_days,
                                                         predict_types=predict_type_in)
        except DataException:
            continue

        with torch.no_grad():
            code_name = get_code_name(stock)
            accuracy, precision, recall, f1 = validate(model, x_test, y_test)
            if sum(y_test) != 0:
                positive_sample = round(sum(y_test).item())
            else:
                positive_sample = 0
            print("stock {} {}: total sample {}, positive sample {}, accuracy {:.2f}%, precision {:.2f}%, "
                  "recall {:.2f}% f1 {:.2f}% "
                  .format(stock, code_name, len(x_test), positive_sample, accuracy, precision, recall, f1))


def save_predict_result(code, model):
    x_test, y_test = construct_dataset_instantly(code, index_list_analysis,
                                                 predict_days=predict_days,
                                                 predict_thresholds=thresholds,
                                                 predict_types=predict_type,
                                                 debug=True)
    model, _, _, _, _ = load(model)
    y_prediction = predict(model, x_test)
    y = y_prediction.cpu().numpy()
    y = pandas.DataFrame(y)
    y.to_csv("temp/" + code + "_result.csv", header=False, index=True)


# save_predict_result("sh.601360", "2020-09-01-19-09-55-60.48-30.11-17.01-model")
validate_model()
