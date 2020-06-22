from stock_query.stock_query import *
from stock_query.stock_utils import *


def prepare_data(day_k_list):
    login()
    for code in day_k_list:
        query_day_k_data(code)

    logout()
