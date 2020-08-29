import time
from os.path import isfile

import torch
from net.net import NeuralNetwork as Net

device = torch.device('cuda:0')
MODEL_DATA_BASE = "model_data/"
MODEL_DATA_PERMANENT_BASE = MODEL_DATA_BASE + "permanent/"


class TrainingParam(object):
    def __init__(self, dataset_type, training_stock_list, test_stock_list, index_list,
                 predict_days, thresholds, predict_type):
        self.dataset_type = dataset_type
        self.training_stock_list = training_stock_list
        self.test_stock_list = test_stock_list
        self.index_list = index_list

        self.x_size = None
        self.net_param = None

        self.predict_days = predict_days
        self.thresholds = thresholds
        self.predict_type = predict_type

    def get_predict_param(self):
        return self.predict_days, self.thresholds, self.predict_type

    def set_x_size(self, x_size):
        self.x_size = x_size

    def set_net_param(self, net_param):
        self.net_param = net_param

    def get_x_size(self):
        return self.x_size

    def get_net_param(self):
        return self.net_param

    def get_type(self):
        return self.dataset_type


def save(model, param, epoch, optimizer, loss,
         val_accuracy, val_precision, val_recall,
         test_accuracy, test_precision, test_recall):
    str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    path = str_time + "-" + str(val_accuracy) + "-" + str(val_precision) + "-" + str(val_recall) + "-model.pt"
    print("save model as:", path)

    torch.save({
        'epoch': epoch,
        'param': param,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val accuracy': val_accuracy,
        'val precision': val_precision,
        'val recall': val_recall,
        'test accuracy': test_accuracy,
        'test precision': test_precision,
        'test recall': test_recall
    }, MODEL_DATA_BASE + path)


def load(path):
    if isfile(MODEL_DATA_PERMANENT_BASE + path):
        whole_path = MODEL_DATA_PERMANENT_BASE + path
    else:
        whole_path = MODEL_DATA_BASE + path
    checkpoint = torch.load(whole_path)
    param = checkpoint['param']
    model = Net(param.get_x_size(), param.get_net_param()).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("load model from", path, "type", param.dataset_type, "total iterations", checkpoint['epoch'])
    try:
        val_accuracy = checkpoint['val accuracy']
        val_precision = checkpoint['val precision']
        val_recall = checkpoint['val recall']
        print("with loss", round(loss.item(), 2),
              "val accuracy", val_accuracy, "precision", val_precision, "recall", val_recall)
    except KeyError:
        print("failed to get validation checkpoint")

    return model, optimizer, epoch, loss, param
