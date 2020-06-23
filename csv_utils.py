from datetime import datetime, timedelta

import numpy
import pandas as pd

STOCK_DATA_DIR_BASE = "stock_data" + "/"
INDIVIDUAL_DIR = STOCK_DATA_DIR_BASE + "individual" + "/"
INDIVIDUAL_SUFFIX = "_day_k_data.csv"
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


def read_individual(code, cols=None):
    ret = pd.read_csv(individual_name(code), usecols=cols, dtype=numpy.float32)
    return ret


def individual_name(code):
    return INDIVIDUAL_DIR + code + INDIVIDUAL_SUFFIX


def get_csv_latest_date(code):
    try:
        df = pd.read_csv(individual_name(code), usecols=['date'])
    except FileNotFoundError:
        print(individual_name(code), "is not found")
        return 0
    except PermissionError:
        print(individual_name(code), "read permission error")
        return 0
    if len(df) == 0:
        print(individual_name(code), "is empty")
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


def getTodayStr():
    return datetime.today().strftime(DATE_PATTERN)
