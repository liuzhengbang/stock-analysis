import glob
import os
from datetime import datetime, timedelta
from os.path import isfile

import numpy
import os
import pandas as pd

STOCK_DATA_DIR_BASE = "stock_data" + "/"
INDIVIDUAL_DIR = STOCK_DATA_DIR_BASE + "individual" + "/"
INDEX_DIR = STOCK_DATA_DIR_BASE + "index" + "/"
SHIBOR_CSV = STOCK_DATA_DIR_BASE + "shibor.csv"
FILTERED_STOCK_FILE = STOCK_DATA_DIR_BASE + "filtered_stock_list.csv"
DAY_K_SUFFIX = "_day_k_data.csv"
DATE_PATTERN = "%Y-%m-%d"

TEMP_DIR_BASE = "temp" + "/"
POSITIVE_CSV = TEMP_DIR_BASE + "positive.csv"
NEGATIVE_CSV = TEMP_DIR_BASE + 'negative.csv'
VAL_POSITIVE_CSV = TEMP_DIR_BASE + "val_positive.csv"
VAL_NEGATIVE_CSV = TEMP_DIR_BASE + "val_negative.csv"

LSTM_CSV_DIR = TEMP_DIR_BASE + "lstm_training" + "/"
LSTM_TRAINING_POSITIVE_CSV_DIR = TEMP_DIR_BASE + "lstm_pos_training" + "/"
LSTM_TRAINING_NEGATIVE_CSV_DIR = TEMP_DIR_BASE + "lstm_neg_training" + "/"
LSTM_VAL_POSITIVE_CSV_DIR = TEMP_DIR_BASE + "lstm_pos_val" + "/"
LSTM_VAL_NEGATIVE_CSV_DIR = TEMP_DIR_BASE + "lstm_neg_val" + "/"


# individual
def individual_name(code):
    return INDIVIDUAL_DIR + code + DAY_K_SUFFIX


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


def read_individual_csv(code, cols=None):
    ret = pd.read_csv(individual_name(code), usecols=cols)
    return ret


# index
def index_name(code):
    return INDEX_DIR + code + DAY_K_SUFFIX


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


def read_index_csv(code, cols=None):
    ret = pd.read_csv(index_name(code), usecols=cols)
    return ret


# shibor
def write_shibor(rs, append=False):
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    if append:
        result.to_csv(SHIBOR_CSV, mode="a", index=False, header=False)
    else:
        result.to_csv(SHIBOR_CSV, index=False)
    return result


# common
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


def get_all_stocks_code_list():
    ret = pd.read_csv(STOCK_DATA_DIR_BASE + "all_stocks.csv", usecols=["code"])
    return ret.values.flatten().tolist()


def get_all_stocks_code_and_name():
    ret = pd.read_csv(STOCK_DATA_DIR_BASE + "all_stocks.csv", usecols=["code", "code_name"], encoding='gbk')
    return ret


def get_stock_by_constituent(stock):
    ret = pd.read_csv(STOCK_DATA_DIR_BASE + stock + "_stocks.csv", usecols=["code"])
    return ret.values.flatten().tolist()


def get_stock_code_list_by_industry(industry_list):
    return_list = []
    ret = pd.read_csv(STOCK_DATA_DIR_BASE + "all_stocks.csv", encoding="gbk")
    for industry in industry_list:
        temp = ret[(ret.industry == industry)]
        return_list = return_list + temp['code'].values.flatten().tolist()
    return return_list


def save_filtered_stock_list(stock_list):
    df = pd.DataFrame(stock_list, columns=["code"])
    df.to_csv(FILTERED_STOCK_FILE, index=False)


def load_filtered_stock_list():
    ret = pd.read_csv(FILTERED_STOCK_FILE, usecols=["code"])
    return ret.values.flatten().tolist()


def save_temp_data(csv_data, columns, file):
    is_exist = isfile(file)
    csv_data.to_csv(file, columns=columns, mode="a", index=False, header=not is_exist)


def save_lstm_temp_data(csv_data, columns, code, temp_dir):
    file = temp_dir + code + ".csv"
    csv_data.to_csv(file, columns=columns, mode="a", index=False)

def save_lstm_list_to_csv(start_list, end_list, code, temp_dir):
    temp_dict = {'start' : start_list, 'end': end_list}
    df = pd.DataFrame(temp_dict)
    file = temp_dir + code + ".csv"
    df.to_csv(file)


def load_temp_data(file):
    csv_data = pd.read_csv(file, dtype=float)
    return csv_data


def load_lstm_temp_data(code, temp_dir, group):
    file = temp_dir + code + ".csv"
    csv_data = pd.read_csv(file, dtype=float)
    return csv_data


def delete_temp_data():
    if isfile(POSITIVE_CSV):
        print("delete positive temp csv file")
        os.remove(POSITIVE_CSV)
    if isfile(NEGATIVE_CSV):
        print("delete negative temp csv file")
        os.remove(NEGATIVE_CSV)
    if isfile(VAL_POSITIVE_CSV):
        print("delete validation positive temp csv file")
        os.remove(VAL_POSITIVE_CSV)
    if isfile(VAL_NEGATIVE_CSV):
        print("delete validation negative temp csv file")
        os.remove(VAL_NEGATIVE_CSV)

    files = glob.glob(LSTM_CSV_DIR + "*")
    for f in files:
        os.remove(f)

    files = glob.glob(LSTM_TRAINING_POSITIVE_CSV_DIR + "*")
    for f in files:
        os.remove(f)

    files = glob.glob(LSTM_TRAINING_NEGATIVE_CSV_DIR + "*")
    for f in files:
        os.remove(f)

    files = glob.glob(LSTM_VAL_POSITIVE_CSV_DIR + "*")
    for f in files:
        os.remove(f)

    files = glob.glob(LSTM_VAL_NEGATIVE_CSV_DIR + "*")
    for f in files:
        os.remove(f)
