from datetime import datetime, timedelta
import random

import torch
import pandas as pd

from utils.consts import device
from utils.csv_utils import read_individual_csv, read_index_csv, delete_temp_data, save_temp_data, NEGATIVE_CSV, \
    POSITIVE_CSV, VAL_POSITIVE_CSV, VAL_NEGATIVE_CSV, load_temp_data
from utils.stock_utils import get_code_name

individual_cols_sel = ['open', 'close', 'amount', 'high', 'low', 'volume',
                       'peTTM', 'pbMRQ']
individual_cols_norm = [10., 10., 100000000., 10., 10., 10000000.,
                        1., 1.]
index_cols_sel = ['open', 'high', 'low', 'close', 'volume', 'amount']
index_cols_norm = [1000., 1000., 1000., 1000., 100000000., 100000000.]

default_rolling_days = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 45, 60, 75,
                        100, 150, 225, 300, 400, 500, 600, 700, 800, 900, 1000]


class DataException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def construct_dataset_instantly(code, index_code_list, predict_days, predict_thresholds, predict_types,
                                val_days=0,
                                history_list=None,
                                debug=False):
    if history_list is None:
        history_list = default_rolling_days

    csv_data, title_list = _prepare_dataset(code, history_list, index_code_list, predict_days,
                                            predict_thresholds, predict_types)

    if val_days != 0:
        csv_data['date'] = pd.to_datetime(csv_data['date'], format='%Y-%m-%d')
        split_date = datetime.today() + timedelta(days=-val_days)
        csv_data = (csv_data[(csv_data.date >= split_date)])
    if debug:
        csv_data.to_csv("temp/" + code + "_temp.csv")

    dataset_x = pd.DataFrame(csv_data, columns=title_list)
    dataset_y = pd.DataFrame(csv_data, columns=['result'])

    return df_to_tensor(dataset_x), df_to_tensor(dataset_y)


def construct_dataset_to_csv(code, index_code_list, predict_days, predict_thresholds, predict_types,
                             history_list=None,
                             val_date_list=None,
                             debug=False):
    if history_list is None:
        history_list = default_rolling_days
    if val_date_list is None:
        val_date_list = []

    csv_data, title_list = _prepare_dataset(code, history_list, index_code_list, predict_days,
                                            predict_thresholds, predict_types)
    if debug:
        csv_data.to_csv("temp/" + code + "_temp.csv")

    _save_temp_data_to_csv_file(csv_data, title_list, val_date_list, debug=debug)


def _prepare_dataset(code, history_list, index_code_list, predict_days, predict_thresholds, predict_types):
    if len(history_list) != 0:
        history_length = max(history_list)
    else:
        history_length = 1
    csv_data = _get_csv_data(code)
    if len(csv_data) < history_length + max(predict_days):
        raise DataException(code)
    csv_data.drop(labels=0, inplace=True)
    _normalize_individual(csv_data)
    title_list = individual_cols_sel.copy()
    csv_data = _append_index(csv_data, index_code_list, title_list)
    if len(csv_data) <= max(predict_days):
        raise DataException(code)
    csv_data = csv_data.reset_index(drop=True)
    csv_data = _append_history(csv_data, title_list, history_list)
    csv_data = _append_predicts(csv_data, predict_days, predict_thresholds, predict_types)
    return csv_data, title_list


def _append_index(csv_data, index_code_list, title_list):
    if len(index_code_list) == 0:
        return

    for index_code in index_code_list:
        index_data = read_index_csv(index_code)
        _normalize_index(index_data)

        csv_data = pd.merge(csv_data, index_data, how="inner", on="date", suffixes=('', '_' + index_code))
        for sel in index_cols_sel:
            title_list.append(sel + '_' + index_code)
    return csv_data


def _get_csv_data(code):
    try:
        csv_data = read_individual_csv(code)
    except FileNotFoundError:
        raise DataException(code)
    except pd.errors.EmptyDataError:
        raise DataException(code)
    csv_data = csv_data[csv_data.tradestatus == 1]
    csv_data = csv_data[csv_data.isST == 0]
    csv_data = csv_data.reset_index(drop=True)

    if csv_data.isna().sum().sum() != 0:
        raise DataException(code)
    return csv_data


def generate_val_data(num_sample, val_recent_days, val_frac=0.2):
    ret = []
    for i in range(num_sample):
        if i < val_recent_days or random.randint(1, 100) < 100 * val_frac:
            date = datetime.today() + timedelta(days=-i)
            ret.append(date.strftime('%Y-%m-%d'))
    return ret


def construct_predict_data(code,
                           index_code_list,
                           history_list=None,
                           ):
    if history_list is None:
        history_list = default_rolling_days

    csv_data = _get_csv_data(code)

    if len(csv_data) < max(history_list):
        raise DataException(code)

    _normalize_individual(csv_data)

    title_list = individual_cols_sel.copy()

    csv_data = _append_index(csv_data, index_code_list, title_list)

    csv_data = csv_data.reset_index(drop=True)
    csv_data = _append_history(csv_data, title_list, history_list)

    csv_data = pd.DataFrame(csv_data).tail(1)
    csv_data = csv_data.reset_index(drop=True)
    predict_data = pd.DataFrame(csv_data, columns=title_list)
    return df_to_tensor(predict_data), csv_data['date'].item()


def df_to_tensor(dataset):
    return torch.tensor(dataset.values).to(device).float()


def _append_history(csv_data, title_list, history_list):
    if len(history_list) == 0:
        return

    assert min(history_list) >= 2
    for days in history_list:
        csv_data['pctChg_' + str(days)] = csv_data['pctChg'].rolling(days).sum()
        title_list.append('pctChg_' + str(days))
        csv_data['volume_' + str(days)] = csv_data['volume'].rolling(days).mean()
        title_list.append('volume_' + str(days))
        csv_data['ma_' + str(days)] = csv_data['close'].rolling(days).mean()
        title_list.append('ma_' + str(days))
        csv_data['highest_' + str(days)] = csv_data['high'].rolling(days).max()
        title_list.append('highest_' + str(days))
        csv_data['lowest_' + str(days)] = csv_data['low'].rolling(days).min()
        title_list.append('lowest_' + str(days))
        csv_data['peTTM_' + str(days)] = csv_data['peTTM'].rolling(days).mean()
        title_list.append('peTTM_' + str(days))
        csv_data['pbMRQ_' + str(days)] = csv_data['pbMRQ'].rolling(days).mean()
        title_list.append('pbMRQ_' + str(days))

    history_length = max(history_list)
    csv_data.drop(labels=range(0, history_length - 1), axis=0, inplace=True)
    csv_data = csv_data.reset_index(drop=True)
    return csv_data


def _append_predicts(csv_data, predict_days, thresholds, predict_type):
    for index in range(len(predict_days)):
        days = predict_days[index]
        threshold = thresholds[index]
        predict = predict_type[index]
        assert predict == "max" or predict == "average"
        if threshold >= 0:
            if predict == "max":
                _add_max_higher_prediction(csv_data, index, days, threshold)
            elif predict == "average":
                _add_average_higher_prediction(csv_data, index, days, threshold)
        else:
            if predict == "max":
                _add_max_lower_prediction(csv_data, index, days, threshold)
            elif predict == "average":
                _add_average_lower_prediction(csv_data, index, days, threshold)

    end_index = len(csv_data)
    drop_days = max(predict_days)
    # print(pd.DataFrame(csv_data, columns=['date', 'result', 'close', 'low', 'avg_chg_3', 'max_chg_1', 'result']))
    csv_data.drop(labels=range(end_index - drop_days, end_index), inplace=True)
    csv_data = csv_data.reset_index(drop=True)
    return csv_data


def _add_max_higher_prediction(csv_data, index, days, threshold):
    csv_data['max_chg_' + str(days)] = \
        (csv_data['high'].rolling(days).max().shift(-days) - csv_data['close']) / csv_data['close']
    if index == 0:
        csv_data['result'] = csv_data.apply(
            lambda x: 1.0 if x['max_chg_' + str(days)] > threshold else 0.0, axis=1)
    else:
        csv_data['result'] = csv_data.apply(
            lambda x: 1.0 if (x['max_chg_' + str(days)] > threshold and x['result'] == 1.0)
            else 0.0, axis=1)


def _add_max_lower_prediction(csv_data, index, days, threshold):
    csv_data['min_chg_' + str(days)] = \
        (csv_data['low'].rolling(days).min().shift(-days) - csv_data['close']) / csv_data['close']
    if index == 0:
        csv_data['result'] = csv_data.apply(
            lambda x: 1.0 if x['min_chg_' + str(days)] < threshold else 0.0, axis=1)
    else:
        csv_data['result'] = csv_data.apply(
            lambda x: 1.0 if (x['min_chg_' + str(days)] < threshold and x['result'] == 1.0)
            else 0.0, axis=1)


def _add_average_higher_prediction(csv_data, index, days, threshold):
    csv_data['avg_h_chg_' + str(days)] = \
        (csv_data['close'].rolling(days).mean().shift(-days) - csv_data['close']) / csv_data['close']
    if index == 0:
        csv_data['result'] = csv_data.apply(
            lambda x: 1.0 if x['avg_h_chg_' + str(days)] > threshold else 0.0, axis=1)
    else:
        csv_data['result'] = csv_data.apply(
            lambda x: 1.0 if (x['avg_h_chg_' + str(days)] > threshold and x['result'] == 1.0)
            else 0.0, axis=1)


def _add_average_lower_prediction(csv_data, index, days, threshold):
    csv_data['avg_l_chg_' + str(days)] = \
        (csv_data['close'].rolling(days).mean().shift(-days) - csv_data['close']) / csv_data['close']
    if index == 0:
        csv_data['result'] = csv_data.apply(
            lambda x: 1.0 if x['avg_l_chg_' + str(days)] < threshold else 0.0, axis=1)
    else:
        csv_data['result'] = csv_data.apply(
            lambda x: 1.0 if (x['avg_l_chg_' + str(days)] < threshold and x['result'] == 1.0)
            else 0.0, axis=1)


def _normalize_individual(frame):
    for index in range(len(individual_cols_sel)):
        frame[individual_cols_sel[index]] = frame[individual_cols_sel[index]] / individual_cols_norm[index]


def _normalize_index(frame):
    for index in range(len(index_cols_sel)):
        frame[index_cols_sel[index]] = frame[index_cols_sel[index]] / index_cols_norm[index]


def _save_temp_data_to_csv_file(csv_data, title_list, val_list, debug=False):
    dataset_training = []
    dataset_val = []
    for index, row in csv_data.iterrows():
        if row['date'] in val_list:
            dataset_val.append(row)
        else:
            dataset_training.append(row)
    dataset_training = pd.DataFrame(dataset_training, columns=csv_data.columns)
    dataset_val = pd.DataFrame(dataset_val, columns=csv_data.columns)

    if debug:
        code = csv_data.code[0]
        dataset_training.to_csv("temp/" + code + "_training.csv")
        dataset_val.to_csv("temp/" + code + "_val.csv")

    pos_train_data = (dataset_training[(dataset_training.result == 1.0)])
    neg_train_data = (dataset_training[(dataset_training.result == 0.0)])
    pos_val_data = (dataset_val[(dataset_val.result == 1.0)])
    neg_val_data = (dataset_val[(dataset_val.result == 0.0)])

    save_temp_data(pos_train_data, title_list, POSITIVE_CSV)
    save_temp_data(neg_train_data, title_list, NEGATIVE_CSV)
    save_temp_data(pos_val_data, title_list, VAL_POSITIVE_CSV)
    save_temp_data(neg_val_data, title_list, VAL_NEGATIVE_CSV)


def construct_temp_csv_data(stock_list, index_code_list,
                            predict_days, thresholds, predict_type,
                            val_date_list):
    delete_temp_data()
    total_stocks = 0
    for code in stock_list:
        try:
            construct_dataset_to_csv(code, index_code_list, val_date_list=val_date_list,
                                     predict_days=predict_days, predict_thresholds=thresholds,
                                     predict_types=predict_type)
            total_stocks = total_stocks + 1
            print(code, get_code_name(code), "processed")
        except DataException:
            print(code, get_code_name(code), "not processed")
    print("total", total_stocks, "stocks constructed")


def load_dataset():
    train_pos_data = load_temp_data(POSITIVE_CSV)
    train_neg_data = load_temp_data(NEGATIVE_CSV)

    val_pos_data = load_temp_data(VAL_POSITIVE_CSV)
    val_neg_data = load_temp_data(VAL_NEGATIVE_CSV)

    total_pos = len(train_pos_data) + len(val_pos_data)
    total_neg = len(train_neg_data) + len(val_neg_data)
    total_sample = total_pos + total_neg
    pos_frac = round(total_pos / total_sample * 100, 2)
    print("pos samples:", total_pos, ", which is", pos_frac, "% of total", total_sample, "samples")
    return train_pos_data, train_neg_data, val_pos_data, val_neg_data
