from datetime import datetime, timedelta
import torch
import pandas as pd
from utils.csv_utils import read_individual_csv, read_index_csv, delete_temp_data, save_temp_data, NEGATIVE_CSV, \
    POSITIVE_CSV, VAL_POSITIVE_CSV, VAL_NEGATIVE_CSV, load_temp_data
from utils.stock_utils import get_code_name

individual_cols_sel = ['open', 'close', 'amount', 'high', 'low', 'volume',
                       'peTTM', 'pbMRQ']
individual_cols_norm = [10., 10., 100000000., 10., 10., 10000000.,
                        1., 1.]
index_cols_sel = ['open', 'high', 'low', 'close', 'volume', 'amount']
index_cols_norm = [1000., 1000., 1000., 1000., 100000000., 100000000.]

device = torch.device('cuda:0')
default_rolling_days = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 45, 60, 75,
                        100, 150, 225, 300, 400, 500, 600, 700, 800, 900, 1000]


class DataException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def construct_dataset(code, index_code_list, predict_days, thresholds, predict_type,
                      val_days=90,
                      append_history=True,
                      append_index=False,
                      rolling_days=None,
                      save_data_to_csv=False,
                      return_data=False,
                      return_only_val_data=False):
    """
    :param return_only_val_data: only return val data for validation
    :param val_days: validation period, from today
    :param save_data_to_csv: whether save data to temp csv for manual investigate
    :param predict_type: max or average
    :param append_history: whether append history data
    :param rolling_days: history days to roll
    :param return_data: whether just return data, without saving to csv
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
    else:
        csv_data = csv_data.reset_index(drop=True)

    if append_history:
        _add_history_data(csv_data, title_list, rolling_days)
        csv_data.drop(labels=range(0, history_length - 1), axis=0, inplace=True)

    csv_data = csv_data.reset_index(drop=True)

    if predict_type == "average":
        _add_average_predicts(csv_data, predict_days, thresholds)
    elif predict_type == "max":
        _add_max_predicts(csv_data, predict_days, thresholds)

    if save_data_to_csv:
        csv_data.to_csv("temp/" + code + "_temp.csv")

    if not return_data:
        _save_temp_data_to_csv_file(csv_data, title_list, val_days)
    else:
        if return_only_val_data:
            csv_data['date'] = pd.to_datetime(csv_data['date'], format='%Y-%m-%d')
            split_date = datetime.today() + timedelta(days=-val_days)
            csv_data = (csv_data[(csv_data.date >= split_date)])
        dataset_x = pd.DataFrame(csv_data, columns=title_list)
        dataset_y = pd.DataFrame(csv_data, columns=['result'])

        return convert_to_tensor(dataset_x), convert_to_tensor(dataset_y)


def construct_predict_data(code, index_code_list,
                           append_history=True,
                           append_index=False,
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

    csv_data = csv_data[csv_data.tradestatus == 1]
    csv_data = csv_data[csv_data.isST == 0]
    csv_data = csv_data.reset_index(drop=True)

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
    else:
        csv_data = csv_data.reset_index(drop=True)

    if append_history:
        _add_history_data(csv_data, title_list, rolling_days)
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


def _normalize_individual(frame):
    for index in range(len(individual_cols_sel)):
        frame[individual_cols_sel[index]] = frame[individual_cols_sel[index]] / individual_cols_norm[index]


def _normalize_index(frame):
    for index in range(len(index_cols_sel)):
        frame[index_cols_sel[index]] = frame[index_cols_sel[index]] / index_cols_norm[index]


def _save_temp_data_to_csv_file(csv_data, title_list, val_days):
    csv_data['date'] = pd.to_datetime(csv_data['date'], format='%Y-%m-%d')

    split_date = datetime.today() + timedelta(days=-val_days)
    pos_train_data = (csv_data[(csv_data.date < split_date) & (csv_data.result == 1.0)])
    neg_train_data = (csv_data[(csv_data.date < split_date) & (csv_data.result == 0.0)])
    pos_val_data = (csv_data[(csv_data.date > split_date) & (csv_data.result == 1.0)])
    neg_val_data = (csv_data[(csv_data.date > split_date) & (csv_data.result == 0.0)])

    save_temp_data(pos_train_data, title_list, POSITIVE_CSV)
    save_temp_data(neg_train_data, title_list, NEGATIVE_CSV)
    save_temp_data(pos_val_data, title_list, VAL_POSITIVE_CSV)
    save_temp_data(neg_val_data, title_list, VAL_NEGATIVE_CSV)


def construct_temp_csv_data(stock_list, index_code_list, predict_days, thresholds, predict_type):
    delete_temp_data()
    for code in stock_list:
        try:
            construct_dataset(code, index_code_list,
                              predict_days=predict_days, thresholds=thresholds, predict_type=predict_type)
            print(code, get_code_name(code), "processed")
        except DataException:
            print(code, get_code_name(code), "not processed")


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
