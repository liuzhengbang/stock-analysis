import torch
import pandas as pd

from utils.csv_utils import read_individual_csv, read_index_csv, save_temp_positive_data, delete_temp_data, \
    save_temp_negative_data
from utils.stock_utils import get_code_name

individual_cols_sel = ['open', 'close', 'amount', 'high', 'low', 'volume',
                       'peTTM', 'pbMRQ']
individual_cols_norm = [10., 10., 100000000., 10., 10., 10000000.,
                        1., 1.]
index_cols_sel = ['open', 'high', 'low', 'close', 'volume', 'amount']
index_cols_norm = [1000., 1000., 1000., 1000., 100000000., 100000000.]

device = torch.device('cuda:0')
# default_rolling_days = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 45, 60, 80, 100, 200, 300,
# 450, 600, 750, 1000]
default_rolling_days = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 45, 60, 75,
                        100, 150, 225, 300, 400, 500, 600, 700, 800, 900, 1000]


class DataException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def construct_dataset(code, index_code_list, predict_days, thresholds, predict_type,
                      append_history=True,
                      append_index=True,
                      rolling_days=None,
                      save_data_to_csv=False,
                      return_data=False):
    """
    :param save_data_to_csv:
    :param predict_type:
    :param append_history:
    :param rolling_days:
    :param thresholds:
    :param return_data:
    :param code: stock code
    :param index_code_list: index code to append
    :param predict_days: predicts n days after the given day, 1 predicts only next day
    :param thresholds: threshold for stock rise indicts positive, 0: is enough 0.1: average 10% rise
    :param append_index: whether append index
    :param return_data: return dataset x and y
    :return:
    """
    assert predict_type == "average" or predict_type == "max"
    if rolling_days is None:
        rolling_days = default_rolling_days
    if append_history:
        history_length = max(rolling_days)
    else:
        history_length = 1

    try:
        csv_data = read_individual_csv(code)
    except FileNotFoundError:
        raise DataException(code)
    except pd.errors.EmptyDataError:
        raise DataException(code)

    csv_data = csv_data[csv_data.tradestatus == 1]
    csv_data = csv_data[csv_data.isST == 0]
    csv_data = csv_data.reset_index(drop=True)

    if len(csv_data) < max(rolling_days) + max(predict_days):
        raise DataException(code)

    csv_data.drop(labels=0, inplace=True)
    _normalize_individual(csv_data)

    title_list = individual_cols_sel.copy()

    if append_index:
        for index_code in index_code_list:
            index_data = read_index_csv(index_code)
            _normalize_index(index_data)

            csv_data = pd.merge(csv_data, index_data, how="inner", on="date", suffixes=('', '_' + index_code))
            for sel in index_cols_sel:
                title_list.append(sel + '_' + index_code)

        if len(csv_data) <= max(predict_days):
            raise DataException(code)

        _add_history_data(csv_data, title_list, rolling_days)
    else:
        csv_data = csv_data.reset_index(drop=True)

    if append_history:
        csv_data.drop(labels=range(0, history_length - 1), axis=0, inplace=True)

    csv_data = csv_data.reset_index(drop=True)

    if predict_type == "average":
        _add_average_predicts(csv_data, predict_days, thresholds)
    elif predict_type == "max":
        _add_max_predicts(csv_data, predict_days, thresholds)

    if save_data_to_csv:
        csv_data.to_csv("temp/" + code + "_temp.csv")

    if not return_data:
        _save_temp_data_to_csv_file(csv_data, title_list)
    else:
        dataset_x = pd.DataFrame(csv_data, columns=title_list)
        dataset_y = pd.DataFrame(csv_data, columns=['result'])

        return convert_to_tensor(dataset_x), convert_to_tensor(dataset_y)


def construct_predict_data(code, index_code_list,
                           append_history=True,
                           append_index=True,
                           rolling_days=None,
                           ):
    if rolling_days is None:
        rolling_days = default_rolling_days
    if append_history:
        history_length = rolling_days[len(rolling_days) - 1]
    else:
        history_length = 1

    try:
        csv_data = read_individual_csv(code)
    except FileNotFoundError:
        raise DataException(code)
    except pd.errors.EmptyDataError:
        raise DataException(code)

    if len(csv_data) < max(rolling_days):
        raise DataException(code)

    _normalize_individual(csv_data)

    title_list = individual_cols_sel.copy()

    if append_index:
        for index_code in index_code_list:
            index_data = read_index_csv(index_code)
            _normalize_index(index_data)

            csv_data = pd.merge(csv_data, index_data, how="inner", on="date", suffixes=('', '_' + index_code))
            for sel in index_cols_sel:
                title_list.append(sel + '_' + index_code)

        _add_history_data(csv_data, title_list, rolling_days)
    else:
        csv_data = csv_data.reset_index(drop=True)

    if append_history:
        csv_data.drop(labels=range(0, history_length - 1), axis=0, inplace=True)

    csv_data = pd.DataFrame(csv_data).tail(1)
    csv_data = csv_data.reset_index(drop=True)
    predict_data = pd.DataFrame(csv_data, columns=title_list)
    return convert_to_tensor(predict_data), csv_data['date'].item()


def convert_to_tensor(dataset):
    return torch.tensor(dataset.values).to(device).float()


def _add_history_data(csv_data, title_list, rolling_days):
    assert min(rolling_days) >= 2
    for days in rolling_days:
        csv_data['pctChg_' + str(days)] = csv_data['pctChg'].rolling(days).sum()
        title_list.append('pctChg_' + str(days))
        csv_data['volume_' + str(days)] = csv_data['volume'].rolling(days).mean()
        title_list.append('volume_' + str(days))


def _add_average_predicts(csv_data, predict_days, thresholds):
    assert min(predict_days) >= 1
    for index in range(len(predict_days)):
        days = predict_days[index]
        threshold = thresholds[index]
        csv_data['avg_chg_' + str(days)] = \
            (csv_data['close'].rolling(days).mean().shift(-days) - csv_data['close']) / csv_data['close']
        if threshold >= 0:
            if index == 0:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if x['avg_chg_' + str(days)] > threshold else 0.0, axis=1)
            else:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if (x['avg_chg_' + str(days)] > threshold and x['result'] == 1.0)
                    else 0.0, axis=1)
        else:
            if index == 0:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if x['avg_chg_' + str(days)] < threshold else 0.0, axis=1)
            else:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if (x['avg_chg_' + str(days)] < threshold and x['result'] == 1.0)
                    else 0.0, axis=1)

    end_index = len(csv_data)
    drop_days = max(predict_days)
    # print(pd.DataFrame(csv_data, columns=['date', 'result', 'close', 'chg_1', 'chg_3']))
    csv_data.drop(labels=range(end_index - drop_days, end_index), inplace=True)


def _add_max_predicts(csv_data, predict_days, thresholds):
    for index in range(len(predict_days)):
        days = predict_days[index]
        threshold = thresholds[index]

        if threshold >= 0:
            csv_data['max_chg_' + str(days)] = \
                (csv_data['high'].rolling(days).max().shift(-days) - csv_data['close']) / csv_data['close']
            if index == 0:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if x['max_chg_' + str(days)] > threshold else 0.0, axis=1)
            else:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if (x['max_chg_' + str(days)] > threshold and x['result'] == 1.0)
                    else 0.0, axis=1)
        else:
            csv_data['min_chg_' + str(days)] = \
                (csv_data['low'].rolling(days).min().shift(-days) - csv_data['close']) / csv_data['close']
            if index == 0:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if x['min_chg_' + str(days)] < threshold else 0.0, axis=1)
            else:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if (x['min_chg_' + str(days)] < threshold and x['result'] == 1.0)
                    else 0.0, axis=1)

    end_index = len(csv_data)
    drop_days = max(predict_days)
    # print(pd.DataFrame(csv_data, columns=['date', 'result', 'close', 'low', 'chg_1', 'chg_3']))
    csv_data.drop(labels=range(end_index - drop_days, end_index), inplace=True)


# def _getDataSet(csv_data, title_list, predict_days, threshold=0.0):
#     csv_data['predict'] = (csv_data['close'].rolling(predict_days).mean().shift(-predict_days) - csv_data['close']) \
#                           / csv_data['close']
#     csv_data['predict_sort'] = csv_data.apply(lambda x: 1.0 if x.predict > threshold else 0.0, axis=1)
#
#     end_index = len(csv_data)
#     csv_data.drop(labels=range(end_index - predict_days, end_index), inplace=True)
#     dataset_x = pd.DataFrame(csv_data, columns=title_list)
#     dataset_y = pd.DataFrame(csv_data, columns=['predict_sort'])
#
#     return dataset_x, dataset_y


def _normalize_individual(frame):
    for index in range(len(individual_cols_sel)):
        frame[individual_cols_sel[index]] = frame[individual_cols_sel[index]] / individual_cols_norm[index]


def _normalize_index(frame):
    for index in range(len(index_cols_sel)):
        frame[index_cols_sel[index]] = frame[index_cols_sel[index]] / index_cols_norm[index]


# def construct_dataset_batch(stock_list, index_code_list):
#     x_train, y_train = construct_dataset(stock_list[0], index_code_list)
#     for code in stock_list[1:]:
#         x_temp, y_temp = construct_dataset(code, index_code_list)
#         x_train = torch.cat([x_train, x_temp], dim=0)
#         y_train = torch.cat([y_train, y_temp], dim=0)
#
#     return x_train, y_train


def _convert_pct_chg_to_bool(pct_chg):
    for value in range(pct_chg.shape[0]):
        if pct_chg[value] > 0:
            pct_chg[value] = 1.
        else:
            pct_chg[value] = 0.


def _save_temp_data_to_csv_file(csv_data, title_list):
    positive_data = csv_data[csv_data.result == 1.0]
    save_temp_positive_data(positive_data, title_list)
    negative_data = csv_data[csv_data.result == 0.0]
    save_temp_negative_data(negative_data, title_list)


def construct_temp_csv_data(stock_list, index_code_list, predict_days, thresholds, predict_type):
    delete_temp_data()
    for code in stock_list:
        try:
            construct_dataset(code, index_code_list,
                              predict_days=predict_days, thresholds=thresholds, predict_type=predict_type)
            print(code, get_code_name(code), "processed")
        except DataException:
            print(code, get_code_name(code), "not processed")
