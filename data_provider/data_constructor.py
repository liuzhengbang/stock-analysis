import torch
import pandas as pd

from utils.csv_utils import read_individual_csv, read_index_csv

individual_cols_sel = ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ']
index_cols_sel = ['open', 'high', 'low', 'close', 'volume', 'amount']
device = torch.device('cuda:0')
rolling_days = [2, 3, 4, 5, 6, 7, 14, 30, 60, 100, 300, 600, 1000]


class DataException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def construct_dataset_with_index(code, index_code_list, filtering_only=False):
    csv_data = read_individual_csv(code)
    csv_data.drop(labels=0, inplace=True)
    title_list = individual_cols_sel.copy()

    for index_code in index_code_list:
        index_data = read_index_csv(index_code)
        csv_data = pd.merge(csv_data, index_data, how="inner", on="date", suffixes=('', '_' + index_code))
        for sel in index_cols_sel:
            title_list.append(sel + '_' + index_code)

    dataset_x = pd.DataFrame(csv_data, columns=title_list)
    dataset_y = pd.DataFrame(csv_data, columns=['pctChg'])

    if len(dataset_x) < 1:
        raise DataException(code)

    if filtering_only:
        return

    dataset_x.drop(labels=len(dataset_x) - 1, inplace=True)
    dataset_y.drop(labels=0, inplace=True)

    dataset_x = torch.tensor(dataset_x.values).to(device).float()
    dataset_y = torch.tensor(dataset_y.values).to(device).float()

    convert_pct_chg_to_bool(dataset_y)
    return dataset_x, dataset_y


def __add_rolling_data__(csv_data, title_list):
    for days in rolling_days:
        csv_data['pctChg_' + str(days)] = csv_data['pctChg'].rolling(days).sum()
        title_list.append('pctChg_' + str(days))
        csv_data['volume_' + str(days)] = csv_data['volume'].rolling(days).sum()
        title_list.append('volume_' + str(days))


def construct_dataset_with_index_and_history(code, index_code_list, filtering_only=False):
    history_length = 1000
    try:
        csv_data = read_individual_csv(code)
    except FileNotFoundError:
        raise DataException(code)
    except pd.errors.EmptyDataError:
        raise DataException(code)

    if len(csv_data) <= 1000:
        raise DataException(code)

    csv_data.drop(labels=0, inplace=True)

    title_list = individual_cols_sel.copy()

    for index_code in index_code_list:
        index_data = read_index_csv(index_code)
        csv_data = pd.merge(csv_data, index_data, how="inner", on="date", suffixes=('', '_' + index_code))
        for sel in index_cols_sel:
            title_list.append(sel + '_' + index_code)

    if len(csv_data) <= 1000:
        raise DataException(code)

    __add_rolling_data__(csv_data, title_list)

    csv_data.drop(labels=range(0, history_length - 1), axis=0, inplace=True)
    csv_data = csv_data.reset_index(drop=True)

    dataset_x = pd.DataFrame(csv_data, columns=title_list)
    dataset_y = pd.DataFrame(csv_data, columns=['pctChg'])

    if filtering_only:
        return

    dataset_x.drop(labels=len(dataset_x) - 1, inplace=True)
    dataset_y.drop(labels=0, inplace=True)

    dataset_x = torch.tensor(dataset_x.values).to(device).float()
    dataset_y = torch.tensor(dataset_y.values).to(device).float()

    convert_pct_chg_to_bool(dataset_y)
    return dataset_x, dataset_y


def construct_dataset_batch(stock_list, index_code_list):
    x_train, y_train = construct_dataset_with_index_and_history(stock_list[0], index_code_list)
    for code in stock_list[1:]:
        x_temp, y_temp = construct_dataset_with_index_and_history(code, index_code_list)
        x_train = torch.cat([x_train, x_temp], dim=0)
        y_train = torch.cat([y_train, y_temp], dim=0)

    return x_train, y_train


def convert_pct_chg_to_bool(pct_chg):
    for value in range(pct_chg.shape[0]):
        if pct_chg[value] > 0:
            pct_chg[value] = 1.
        else:
            pct_chg[value] = 0.
