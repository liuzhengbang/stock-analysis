# coding=gbk
from utils.csv_utils import get_all_stocks_code_and_name, get_stock_code_list_by_industry, get_stock_by_constituent

df = get_all_stocks_code_and_name()


def get_code_name_list(code_list):
    ret = []
    for code in code_list:
        ret.append(get_code_name(code))
    return ret


def get_code_name(code):
    ret = df[df["code"] == code]["code_name"].values[0]
    return ret


def stock_code_list_by_industry_in_constituent(industry, selected_set):
    industry_set = []
    if len(industry) != 0:
        industry_set = get_stock_code_list_by_industry(industry)
    if len(selected_set) == 0:
        return industry_set
    stock_set = set()
    for temp in selected_set:
        stock_list_in_set = get_stock_by_constituent(temp)
        stock_set = set(stock_list_in_set) | stock_set
    if len(industry) != 0:
        ret = _merge_list(industry_set, stock_set)
    else:
        ret = stock_set
    ret = list(ret)
    ret.sort()
    return ret


def _merge_list(stock_list_1, stock_list_2):
    set_1 = set(stock_list_1)
    set_2 = set(stock_list_2)
    ret = set_1 & set_2
    ret = list(ret)
    ret.sort()
    return list(ret)
