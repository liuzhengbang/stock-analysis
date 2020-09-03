# coding=gbk
import pandas
import torch

from data_provider.data_constructor import construct_dataset, DataException
from net.model import load
from net.trainer import validate, predict
from utils.consts import index_list_analysis
from utils.csv_utils import get_stock_code_list_by_industry
from utils.stock_utils import get_stock_code_list_of_industry_contained_in_selected_set, get_code_name

predict_days = [15]
thresholds = [0.1]
predict_type = ["max"]


def _validate_model_with_stock_list(model_name, stock_list, index_list_analysis_in, validate_with_not_training_data,
                                    val_days,
                                    predict_days_in, thresholds_in, predict_type_in):
    model, _, _, _, _ = load(model_name)
    for stock in stock_list:
        try:
            x_test, y_test = construct_dataset(stock, index_list_analysis_in,
                                               predict_days=predict_days_in, thresholds=thresholds_in,
                                               val_days=val_days,
                                               predict_type=predict_type_in,
                                               return_data=True,
                                               return_only_val_data=validate_with_not_training_data)
        except DataException:
            continue

        with torch.no_grad():
            code_name = get_code_name(stock)
            accuracy, precision, recall = validate(model, x_test, y_test)
            if sum(y_test) != 0:
                positive_sample = round(sum(y_test).item())
            else:
                positive_sample = 0
            print("stock {} {}: total sample {}, positive sample {}, accuracy {}%, precision {}%, recall {}%"
                  .format(stock, code_name, len(x_test), positive_sample, accuracy, precision, recall))


def validate_model():
    validate_with_not_training_data = True
    all_stock_list = get_stock_code_list_of_industry_contained_in_selected_set(
        ["银行"], [])
    print(all_stock_list)
    _validate_model_with_stock_list("2020-09-04-00-20-22-99.63-93.83-94.41-model",
                                    all_stock_list, index_list_analysis, validate_with_not_training_data, val_days=45,
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


# save_predict_result("sh.601360", "2020-09-01-19-09-55-60.48-30.11-17.01-model")
validate_model()
