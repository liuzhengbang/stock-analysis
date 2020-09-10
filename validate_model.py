# coding=gbk
import pandas
import torch
import pandas as pd

from dataset.data_constructor import construct_dataset_instantly, DataException
from model.model_persistence import load, get_prediction_from_param
from model.trainer import validate, predict
from utils.consts import index_list_analysis
from utils.csv_utils import get_stock_code_list_by_industry
from utils.stock_utils import stock_code_list_by_industry_in_constituent, get_code_name, get_code_name_list


def validate_model():
    predict_thresholds = [0.1, 0.05]
    predict_days = [6, 15]
    predict_types = ["max", "average"]
    days = 100

    model, _, _, _, _, param = load("20200909-061343-98.0-84.0-model@sz50-pos")
    predict_days = param.predict_days
    predict_thresholds = param.predict_thresholds
    predict_types = param.predict_types

    industry_list = param.get_industry_list()
    select_set = param.get_constituent()

    all_stock_list = stock_code_list_by_industry_in_constituent(
        industry_list, select_set)
    print(all_stock_list)
    _validate_model_with_stock_list(model,
                                    stock_list=all_stock_list, index_list_analysis_in=index_list_analysis,
                                    val_days=days,
                                    predict_days_in=predict_days,
                                    thresholds_in=predict_thresholds,
                                    predict_type_in=predict_types)


def _validate_model_with_stock_list(model, stock_list, index_list_analysis_in,
                                    val_days,
                                    predict_days_in,
                                    thresholds_in,
                                    predict_type_in):
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
                  "recall {:.2f}%, f1 {:.2f}% "
                  .format(stock, code_name, len(x_test), positive_sample, accuracy, precision, recall, f1))


def save_predict_result(code, model):
    model, _, _, _, _, param = load(model)
    predict_days = param.predict_days
    predict_thresholds = param.predict_thresholds
    predict_types = param.predict_types
    x_test, y_test = construct_dataset_instantly(code, index_list_analysis,
                                                 predict_days=predict_days,
                                                 predict_thresholds=predict_thresholds,
                                                 predict_types=predict_types,
                                                 val_days=0,
                                                 debug=True)

    stock_list = stock_code_list_by_industry_in_constituent(param.get_industry_list()
                                                            , param.get_constituent())
    print("stock list:", stock_list)
    print("stock name list:", get_code_name_list(stock_list))
    print(get_prediction_from_param(param))

    accuracy, precision, recall, f1 = validate(model, x_test, y_test)
    code_name = get_code_name(code)
    if sum(y_test) != 0:
        positive_sample = round(sum(y_test).item())
    else:
        positive_sample = 0
    print("stock {} {}: total sample {}, positive sample {}, accuracy {:.2f}%, precision {:.2f}%, "
          "recall {:.2f}%, f1 {:.2f}% "
          .format(code, code_name, len(x_test), positive_sample, accuracy, precision, recall, f1))

    y_prediction = predict(model, x_test)
    y_prediction = y_prediction.cpu().numpy()
    y_prediction = pandas.DataFrame(y_prediction)
    y_prediction.columns = ["predict"]

    data = pd.read_csv("temp/" + code + "_temp.csv")
    assert len(y_prediction) == len(data)
    data["predict"] = y_prediction.predict - 1
    cols = ["date", "close", "result", "predict"]

    for i in range(len(predict_days)):
        if predict_thresholds[i] >= 0 and predict_types[i] == "max":
            cols.append("max_chg_" + str(predict_days[i]))
        elif predict_thresholds[i] >= 0 and predict_types[i] == "average":
            cols.append("avg_h_chg_" + str(predict_days[i]))
        elif predict_thresholds[i] < 0 and predict_types[i] == "max":
            cols.append("max_chg_" + str(predict_days[i]))
        elif predict_thresholds[i] < 0 and predict_types[i] == "average":
            cols.append("avg_l_chg_" + str(predict_days[i]))

    data = pd.DataFrame(data, columns=cols)
    data.to_csv("temp/" + code + "_temp.csv", index=False)


save_predict_result("sh.600196", "20200909-061343-98.0-84.0-model@sz50-pos")
# validate_model()
