import torch
import pandas as pd

from utils.csv_utils import read_individual_csv, read_index_csv

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


def construct_dataset(code, index_code_list, predict_days=1, threshold=0.0, append_index=True, append_history=True,
                      filtering_only=False):
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

    if len(csv_data) <= history_length + predict_days:
        raise DataException(code)

    csv_data.drop(labels=0, inplace=True)
    __normalize_individual__(csv_data)

    title_list = individual_cols_sel.copy()

    if append_index:
        for index_code in index_code_list:
            index_data = read_index_csv(index_code)
            __normalize_index__(index_data)

            csv_data = pd.merge(csv_data, index_data, how="inner", on="date", suffixes=('', '_' + index_code))
            for sel in index_cols_sel:
                title_list.append(sel + '_' + index_code)

        if len(csv_data) <= predict_days:
            raise DataException(code)

        __add_rolling_data__(csv_data, title_list)
    else:
        csv_data = csv_data.reset_index(drop=True)

    if append_history:
        csv_data.drop(labels=range(0, history_length - 1), axis=0, inplace=True)

    csv_data = csv_data.reset_index(drop=True)

    dataset_x, dataset_y = __getDataSet__(csv_data, title_list, predict_days, threshold=threshold)

    if filtering_only:
        print(code, "has", len(dataset_x), "samples with positive percentage",
              round(dataset_y['predict_sort'].sum()/len(dataset_x) * 100, 2), "%")
        return

    dataset_x = __convert_to_tensor__(dataset_x)
    dataset_y = __convert_to_tensor__(dataset_y)

    return dataset_x, dataset_y


def __convert_to_tensor__(dataset):
    return torch.tensor(dataset.values).to(device).float()


def __add_rolling_data__(csv_data, title_list):
    for days in rolling_days:
        csv_data['pctChg_' + str(days)] = csv_data['pctChg'].rolling(days).sum()
        title_list.append('pctChg_' + str(days))
        csv_data['volume_' + str(days)] = csv_data['volume'].rolling(days).sum()
        title_list.append('volume_' + str(days))


def __getDataSet__(csv_data, title_list, predict_days, threshold=0.0):
    csv_data['predict'] = (csv_data['close'].rolling(predict_days).mean().shift(-predict_days) - csv_data['close']) \
                          / csv_data['close']
    csv_data['predict_sort'] = csv_data.apply(lambda x: 1.0 if x.predict > threshold else 0.0, axis=1)

    end_index = len(csv_data)
    csv_data.drop(labels=range(end_index - predict_days, end_index), inplace=True)
    dataset_x = pd.DataFrame(csv_data, columns=title_list)
    dataset_y = pd.DataFrame(csv_data, columns=['predict_sort'])

    return dataset_x, dataset_y


def __normalize_individual__(frame):
    for index in range(len(individual_cols_sel)):
        frame[individual_cols_sel[index]] = frame[individual_cols_sel[index]] / individual_cols_norm[index]


def __normalize_index__(frame):
    for index in range(len(index_cols_sel)):
        frame[index_cols_sel[index]] = frame[index_cols_sel[index]] / index_cols_norm[index]


def construct_dataset_batch(stock_list, index_code_list):
    x_train, y_train = construct_dataset(stock_list[0], index_code_list)
    for code in stock_list[1:]:
        x_temp, y_temp = construct_dataset(code, index_code_list)
        x_train = torch.cat([x_train, x_temp], dim=0)
        y_train = torch.cat([y_train, y_temp], dim=0)

    return x_train, y_train


def __convert_pct_chg_to_bool__(pct_chg):
    for value in range(pct_chg.shape[0]):
        if pct_chg[value] > 0:
            pct_chg[value] = 1.
        else:
            pct_chg[value] = 0.
