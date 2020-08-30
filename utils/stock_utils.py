# coding=gbk
from utils.csv_utils import get_all_stocks_code_and_name, get_stock_code_list_by_industry, get_stock_code_list

df = get_all_stocks_code_and_name()


def get_code_name_list(code_list):
    ret = []

    for code in code_list:
        ret.append(get_code_name(code))
    return ret


def get_code_name(code):
    ret = df[df["code"] == code]["code_name"].values[0]
    return ret


def get_industry_code_list_in_code_set(industry, stock_set):
    stock_list = get_stock_code_list_by_industry(industry)
    stock_list_in_set = get_stock_code_list(stock_set)
    return _merge_list(stock_list, stock_list_in_set)


def _merge_list(stock_list_1, stock_list_2):
    set_1 = set(stock_list_1)
    set_2 = set(stock_list_2)
    ret = set_1 & set_2
    ret = list(ret)
    ret.sort()
    return list(ret)
