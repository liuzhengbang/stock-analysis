from datetime import datetime, timedelta

import numpy
import pandas as pd

STOCK_DATA_DIR_BASE = "stock_data" + "/"
INDIVIDUAL_DIR = STOCK_DATA_DIR_BASE + "individual" + "/"
INDEX_DIR = STOCK_DATA_DIR_BASE + "index" + "/"
FILTERED_STOCK_FILE = STOCK_DATA_DIR_BASE + "filtered_stock_list.csv"
DAY_K_SUFFIX = "_day_k_data.csv"
DATE_PATTERN = "%Y-%m-%d"


def write_individual(code, rs, append=False):
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    if append:
        result.to_csv(individual_name(code), mode="a", index=False, header=False)
    else:
        result.to_csv(individual_name(code), index=False)
    # print(result)
    return result


def write_index(code, rs, append=False):
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    if append:
        result.to_csv(index_name(code), mode="a", index=False, header=False)
    else:
        result.to_csv(index_name(code), index=False)
    return result


def read_individual_csv(code, cols=None):
    ret = pd.read_csv(individual_name(code), usecols=cols)
    return ret


def read_index_csv(code, cols=None):
    ret = pd.read_csv(index_name(code), usecols=cols)
    return ret


def individual_name(code):
    return INDIVIDUAL_DIR + code + DAY_K_SUFFIX


def index_name(code):
    return INDEX_DIR + code + DAY_K_SUFFIX


def get_csv_latest_date(csv_name):
    try:
        df = pd.read_csv(csv_name, usecols=['date'])
    except FileNotFoundError:
        # print(csv_name, "is not found")
        return 0
    except PermissionError:
        print(csv_name, "read permission error")
        return 0
    if len(df) == 0:
        print(csv_name, "is empty")
        return 0

    # start = df.loc[0, "date"]
    end = df.loc[len(df) - 1, "date"]
    return end


def get_next_day_str(date_str):
    date = convert_str_to_date(date_str)
    one_day = timedelta(days=1)
    next_day = date + one_day
    return next_day.strftime(DATE_PATTERN)


def convert_str_to_date(date_str):
    return datetime.strptime(date_str, DATE_PATTERN)


def get_today_str():
    return datetime.today().strftime(DATE_PATTERN)


def save_stock_code_to_csv(rs, name):
    data = []
    while (rs.error_code == '0') & rs.next():
        data.append(rs.get_row_data())
    result = pd.DataFrame(data, columns=rs.fields)
    result.to_csv(STOCK_DATA_DIR_BASE + name + "_stocks.csv", encoding="gbk", index=False)


def get_all_stocks_code():
    ret = pd.read_csv(STOCK_DATA_DIR_BASE + "all_stocks.csv", usecols=["code"])
    return ret.values.flatten().tolist()


def save_filtered_stock_list(stock_list):
    df = pd.DataFrame(stock_list, columns=["code"])
    df.to_csv(FILTERED_STOCK_FILE, index=False)


def load_filtered_stock_list():
    ret = pd.read_csv(FILTERED_STOCK_FILE, usecols=["code"])
    return ret.values.flatten().tolist()

