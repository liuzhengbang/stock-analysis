import torch
import pandas as pd

from csv_utils import read_individual_csv, read_index_csv

individual_cols_sel = ['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ']
index_cols_sel = ['open', 'high', 'low', 'close', 'volume', 'amount']
device = torch.device('cuda:0')


def construct_x(code):
    dataset_x = read_individual_csv(code, cols=individual_cols_sel)
    dataset_x.drop(labels=len(dataset_x) - 1, inplace=True)
    dataset_x = torch.tensor(dataset_x.values)
    return dataset_x


def construct_dataset_with_index(code, index_code_list, shift=1):
    csv_data = read_individual_csv(code)
    title_list = individual_cols_sel.copy()

    for index_code in index_code_list:
        index_data = read_index_csv(index_code)
        csv_data = pd.merge(csv_data, index_data, how="inner", on="date", suffixes=('', '_' + index_code))
        for sel in index_cols_sel:
            title_list.append(sel + '_' + index_code)

    dataset_x = pd.DataFrame(csv_data, columns=title_list)
    dataset_y = pd.DataFrame(csv_data, columns=['pctChg'])

    dataset_x.drop(labels=len(dataset_x) - shift, inplace=True)
    dataset_y.drop(labels=0, inplace=True)

    dataset_x = torch.tensor(dataset_x.values).to(device).float()
    dataset_y = torch.tensor(dataset_y.values).to(device).float()

    convertPctChgToBool(dataset_y)
    return dataset_x, dataset_y


def construct_y(code):
    dataset_y = read_individual_csv(code, cols=['pctChg'])
    dataset_y.drop(labels=0, inplace=True)
    dataset_y = torch.tensor(dataset_y.values)
    convertPctChgToBool(dataset_y)
    return dataset_y


def convertPctChgToBool(pct_chg):
    for value in range(pct_chg.shape[0]):
        if pct_chg[value] > 0:
            pct_chg[value] = 1.
        else:
            pct_chg[value] = 0.


