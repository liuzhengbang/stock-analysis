from stock_query.stock_query import *
from stock_query.stock_utils import *


def prepare_data(day_k_list, append=True):
    append = False
    login()
    for code in day_k_list:
        query_day_k_data(code, append=append)

    logout()
