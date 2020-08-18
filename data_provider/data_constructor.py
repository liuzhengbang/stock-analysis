import torch
import pandas as pd

from utils.csv_utils import read_individual_csv, read_index_csv, save_temp_positive_data, delete_temp_data, \
    save_temp_negative_data

individual_cols_sel = ['open', 'close', 'amount', 'high', 'low', 'volume',
                       'peTTM', 'pbMRQ']
individual_cols_norm = [10., 10., 100000000., 10., 10., 10000000.,
                        1., 1.]
index_cols_sel = ['open', 'high', 'low', 'close', 'volume', 'amount']
index_cols_norm = [1000., 1000., 1000., 1000., 100000000., 100000000.]

device = torch.device('cuda:0')
rolling_days = [2, 3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 60, 100, 300, 600, 1000]


class DataException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def construct_dataset(code, index_code_list, predict_days=None, thresholds=None, append_index=True, append_history=True,
                      return_data=False):
    """

    :param thresholds:
    :param return_data:
    :param code: stock code
    :param index_code_list: index code to append
    :param predict_days: predicts n days after the given day, 1 predicts only next day
    :param thresholds: threshold for stock rise indicts positive, 0: is enough 0.1: average 10% rise
    :param append_index: whether append index
    :param append_history: whether append history
    :param return_data: return dataset x and y
    :return:
    """

    if thresholds is None:
        thresholds = [0.05]
    if predict_days is None:
        predict_days = [5]
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

    if len(csv_data) <= history_length + max(predict_days):
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

        _add_history_data(csv_data, title_list)
    else:
        csv_data = csv_data.reset_index(drop=True)

    if append_history:
        csv_data.drop(labels=range(0, history_length - 1), axis=0, inplace=True)

    csv_data = csv_data.reset_index(drop=True)

    _add_predicts_days(csv_data, predict_days, thresholds)

    if not return_data:
        _save_temp_data_to_csv_file(csv_data, title_list)
    else:
        dataset_x = pd.DataFrame(csv_data, columns=title_list)
        dataset_y = pd.DataFrame(csv_data, columns=['result'])

        return convert_to_tensor(dataset_x), convert_to_tensor(dataset_y)
    # # dataset_x = __convert_to_tensor__(dataset_x)
    # # dataset_y = __convert_to_tensor__(dataset_y)
    #
    # return dataset_x, dataset_y


def convert_to_tensor(dataset):
    return torch.tensor(dataset.values).to(device).float()


def _add_history_data(csv_data, title_list):
    for days in rolling_days:
        csv_data['pctChg_' + str(days)] = csv_data['pctChg'].rolling(days).sum()
        title_list.append('pctChg_' + str(days))
        csv_data['volume_' + str(days)] = csv_data['volume'].rolling(days).sum()
        title_list.append('volume_' + str(days))


def _add_predicts_days(csv_data, predict_days, thresholds):
    for index in range(len(predict_days)):
        days = predict_days[index]
        threshold = thresholds[index]
        csv_data['predict' + str(days)] = \
            (csv_data['close'].rolling(days).mean().shift(-days) - csv_data['close'])/csv_data['close']
        if threshold > 0:
            if index == 0:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if x['predict' + str(days)] > threshold else 0.0, axis=1)
            else:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if (x['predict' + str(days)] > threshold and csv_data['result'] == 1.0)
                    else 0.0, axis=1)
        else:
            if index == 0:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if x['predict' + str(days)] < threshold else 0.0, axis=1)
            else:
                csv_data['result'] = csv_data.apply(
                    lambda x: 1.0 if (x['predict' + str(days)] < threshold and csv_data['result'] == 1.0)
                    else 0.0, axis=1)

    end_index = len(csv_data)
    drop_days = max(predict_days)
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


def construct_temp_csv_data(stock_list, index_code_list):
    delete_temp_data()
    for code in stock_list:
        try:
            construct_dataset(code, index_code_list)
            print(code, "processed")
        except DataException:
            print(code, "not processed")
