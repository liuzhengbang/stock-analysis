import torch
from torch.utils.data import Dataset
import numpy as np


class LSTMTrainingDataset(Dataset):
    def __init__(self, code_list, pos_list, neg_list):
        self.code_list = code_list
        self.pos_list = pos_list
        self.neg_list = neg_list

        self.pos_length = sum(self.pos_list)
        self.neg_length = sum(self.neg_list)

        self.length = max(self.pos_length, self.neg_length) * 2

    def __getitem__(self, ndx):
        index = ndx // 2
        if ndx % 2 == 0:
            pos_index = index % self.pos_length
            return self.pos_data.values[pos_index], torch.tensor([1.0])
        else:
            neg_index = index % self.neg_length
            return self.neg_data.values[neg_index], torch.tensor([0.0])

    def __len__(self):
        return self.length


def _get_data_position(index, length_list, code_list):
    irr = 0
    for i in range(len(length_list)):
        irr = irr + length_list[i]
        if irr > index:
            return code_list[i], index - (irr - length_list[i])


def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler



def train_model(print_cost=False):
    n = 100
    sched = cosine(n)
    lrs = [sched(t, 1) for t in range(n * 4)]

    # trn_ds, val_ds, enc = create_datasets(x_trn, y_trn['surface'])
