import torch
import numpy as np
import pandas as pd

from csv_utils import read_individual
from stock_query.stock_query import query_individual_day_k_data


def get_x():
    # ori = query_day_k_data("sh.000935")
    # stage1 = np.array(ori)
    # print(stage1.size)
    # output = torch.from_numpy(stage1[[4], [4]])
    output = pd.read_csv('sh.601668_day_k_data.csv', usecols=['tradestatus', 'close'])
    output = torch.tensor(output.values)
    return output


def construct_train_x(code):
    dataset_x = read_individual(code,
                                cols=['open', 'close', 'amount', 'high', 'low', 'volume', 'peTTM', 'pbMRQ'])
    dataset_x.drop(labels=len(dataset_x)-1, inplace=True)
    dataset_x = torch.tensor(dataset_x.values)
    return dataset_x


def construct_train_y(code):
    dataset_y = read_individual(code, cols=['pctChg'])  # (4960, 1)
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


