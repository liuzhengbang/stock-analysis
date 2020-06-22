import torch
import numpy as np
import pandas as pd
from stock_query.stock_query import query_day_k_data


def get_x():
    # ori = query_day_k_data("sh.000935")
    # stage1 = np.array(ori)
    # print(stage1.size)
    # output = torch.from_numpy(stage1[[4], [4]])
    output = pd.read_csv('sh.601668_day_k_data.csv', usecols=['tradestatus', 'close'])
    output = torch.Tensor(output.values)
    print("************************")
    print(output)
    return output


def get_y():
    output = pd.read_csv('sh.601668_day_k_data.csv', usecols=['tradestatus'])
    output = torch.Tensor(output.values)
    print("************************1")
    print(output)
    return output


