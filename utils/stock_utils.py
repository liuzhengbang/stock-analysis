# coding=gbk
from utils.consts import DATE_FORMAT
from utils.csv_utils import get_all_stocks_code_and_name, get_stock_code_list_by_industry, get_stock_by_constituent
from datetime import datetime, timedelta
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


def complete_val_date_list(val_list):
    delta = timedelta(days=365 * 10)
    today = datetime.today()
    recent_date = today
    for date_str in val_list:
        val_date = datetime.strptime(date_str, DATE_FORMAT)
        delta_temp = today - val_date
        if delta_temp < delta:
            delta = delta_temp
            recent_date = val_date

    print("most recent validation date is", recent_date.strftime(DATE_FORMAT))
    days_delta = (today - recent_date).days

    for i in range(days_delta):
        date = datetime.today() + timedelta(days=-i)
        val_list.append(date.strftime(DATE_FORMAT))
        print("add", date.strftime(DATE_FORMAT), "to validation list")

    val_length = 1
    for i in range(len(val_list)):
        date = datetime.today() + timedelta(days=-i)
        date_str = date.strftime(DATE_FORMAT)
        if date_str not in val_list:
            val_length = i
            break
    val_length = val_length - 1
    print("validation length is", val_length - days_delta)

    for i in range(days_delta):
        date = datetime.today() + timedelta(days=-val_length+i)
        date_str = date.strftime(DATE_FORMAT)
        if date_str in val_list:
            val_list.remove(date_str)
            print("remove", date_str, "from validation list")

    return val_list


def get_validation_length(val_list):
    delta = timedelta(days=365 * 10)
    today = datetime.today()
    recent_date = today
    val_list = val_list.copy()
    for date_str in val_list:
        val_date = datetime.strptime(date_str, DATE_FORMAT)
        delta_temp = today - val_date
        if delta_temp < delta:
            delta = delta_temp
            recent_date = val_date

    days_delta = (today - recent_date).days

    for i in range(days_delta):
        date = datetime.today() + timedelta(days=-i)
        val_list.append(date.strftime(DATE_FORMAT))

    val_length = 1
    for i in range(len(val_list)):
        date = datetime.today() + timedelta(days=-i)
        date_str = date.strftime(DATE_FORMAT)
        if date_str not in val_list:
            val_length = i
            break
    val_length = val_length - 1
    print("validation starts from", datetime.today() + timedelta(days=-val_length))
    print("validation length is", val_length - days_delta)

    return val_length
