# coding=gbk
from net.trainer import validate_model
from utils.consts import index_list_analysis
from utils.stock_utils import get_industry_code_list_in_code_set

predict_days = [6]
thresholds = [0.1]
predict_type = "max"


def validate():
    all_stock_list = get_industry_code_list_in_code_set(["通信", "电子"], "hs300")

    validate_model("2020-08-29-23-22-22-76.69-30.6-9.91-model.pt",
                   all_stock_list, index_list_analysis,
                   predict_days=predict_days, thresholds=thresholds, predict_type=predict_type)


validate()
