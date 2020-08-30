from stock_query.stock_query import *
from stock_query.stock_query_helper import *


def prepare_data(individual_day_k_list, index_day_k_list, append=True):
    login()

    query_stock_code()
    query_shibor(append=append)

    for code in individual_day_k_list:
        query_individual_day_k_data(code, append=append)

    for code in index_day_k_list:
        query_index_day_k_data(code, append=append)

    logout()
