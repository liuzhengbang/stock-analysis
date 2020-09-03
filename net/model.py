import time
from os.path import isfile

import torch
from net.net import NeuralNetwork as Net
from utils.consts import device

MODEL_DATA_BASE = "model_data/"
MODEL_DATA_PERMANENT_BASE = MODEL_DATA_BASE + "permanent/"
MODEL_SUFFIX = ".pt"


class TrainingParam(object):
    def __init__(self, industry_list, selected_set, training_stock_list, test_stock_list, index_list,
                 predict_days, thresholds, predict_type):
        self.industry_list = industry_list
        self.training_stock_list = training_stock_list
        self.test_stock_list = test_stock_list
        self.index_list = index_list
        self.selected_set = selected_set

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
        return self.industry_list

    def get_training_stock_list(self):
        return self.training_stock_list

    def get_industry_list(self):
        return self.industry_list

    def get_select_set(self):
        return self.selected_set


def save(model, param, epoch, optimizer, batch_size, loss,
         val_accuracy, val_precision, val_recall,
         test_accuracy, test_precision, test_recall):
    str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    path = str_time + "-" + "%.2f" % val_accuracy \
           + "-" + "%.2f" % val_precision \
           + "-" + "%.2f" % val_recall \
           + "-model"
    print("save model as:", path)

    torch.save({
        'epoch': epoch,
        'param': param,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_size': batch_size,
        'loss': loss,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall
    }, MODEL_DATA_BASE + path + MODEL_SUFFIX)


def load(path):
    if isfile(MODEL_DATA_PERMANENT_BASE + path + MODEL_SUFFIX):
        whole_path = MODEL_DATA_PERMANENT_BASE + path + MODEL_SUFFIX
    else:
        whole_path = MODEL_DATA_BASE + path + MODEL_SUFFIX
    checkpoint = torch.load(whole_path)
    param = checkpoint['param']
    model = Net(param.get_x_size(), param.get_net_param()).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print("load model from", path, "type", param.industry_list, "total iterations", checkpoint['epoch'])
    print("model predict days", param.predict_days, "threshold", param.thresholds, "predict type", param.predict_type)
    try:
        val_accuracy = checkpoint['val_accuracy']
        val_precision = checkpoint['val_precision']
        val_recall = checkpoint['val_recall']
        print("with loss", round(loss.item(), 2),
              "val accuracy", val_accuracy, "precision", val_precision, "recall", val_recall)
    except KeyError:
        print("failed to get validation checkpoint")

    return model, optimizer, epoch, loss, param


def get_prediction_from_param(param):
    predict_days, thresholds, predict_type = param.get_predict_param()
    assert len(predict_days) == len(thresholds)
    ret = "PREDICTIONS: stock "

    for i in range(len(predict_days)):
        threshold = str(thresholds[i] * 100) + "%"
        day = str(predict_days[i])
        if thresholds[i] >= 0 and predict_type[i] == "max":
            ret = ret + "raise to highest [" + threshold + "] in [" + day + "] days"
        elif thresholds[i] < 0 and predict_type[i] == "max":
            ret = ret + "drop to lowest [" + threshold + "] in [" + day + "] days"
        elif thresholds[i] >= 0 and predict_type[i] == "average":
            ret = ret + "avg price will be [" + threshold + "] higher in [" + day + "] days"
        elif thresholds[i] < 0 and predict_type[i] == "average":
            ret = ret + "avg price will be [" + threshold + "] lower in [" + day + "] days"

        if i != len(predict_days) - 1:
            ret = ret + ","
    return ret
