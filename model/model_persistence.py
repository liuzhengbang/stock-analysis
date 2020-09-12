import time
from os.path import isfile

import torch
from model.net import NeuralNetwork as Net
from utils.consts import device

MODEL_DATA_BASE = "model_data/"
MODEL_DATA_PERMANENT_BASE = MODEL_DATA_BASE + "permanent/"
MODEL_SUFFIX = ".pt"


class TrainingParam(object):
    def __init__(self, industry_list, constituent_list, training_stock_list, test_stock_list, index_code_list,
                 val_date_list,
                 predict_days, predict_thresholds, predict_types):
        self.industry_list = industry_list
        self.training_stock_list = training_stock_list
        self.test_stock_list = test_stock_list
        self.index_code_list = index_code_list
        self.constituent_list = constituent_list
        self.val_date_list = val_date_list

        self.net_input_size = None
        self.net_layers = None

        self.predict_days = predict_days
        self.predict_thresholds = predict_thresholds
        self.predict_types = predict_types

    def get_predict_param(self):
        return self.predict_days, self.predict_thresholds, self.predict_types

    def set_net_input_size(self, x_size):
        self.net_input_size = x_size

    def set_net_layers(self, net_param):
        self.net_layers = net_param

    def get_net_input_size(self):
        return self.net_input_size

    def get_net_layers(self):
        return self.net_layers

    def get_training_stock_list(self):
        return self.training_stock_list

    def get_index_code_list(self):
        return self.index_code_list

    def get_test_stock_list(self):
        return self.test_stock_list

    def get_industry_list(self):
        return self.industry_list

    def get_constituent(self):
        return self.constituent_list

    def get_val_date_list(self):
        return self.val_date_list

    def set_val_date_list(self, date_list):
        self.val_date_list = date_list


def save(model, param, epoch, optimizer, batch_size, loss,
         val_accuracy, val_precision, val_recall, val_f1,
         test_accuracy, test_precision, test_recall, test_f1):
    str_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    persistence_file_name = str_time + "-" + "%.1f" % val_accuracy \
                            + "-" + "%.1f" % val_precision \
                            + "-model"
    print("save model as:\033[35m", persistence_file_name, "\033[0m")

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
        'val_f1': val_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }, MODEL_DATA_BASE + persistence_file_name + MODEL_SUFFIX)


def load(path):
    if isfile(MODEL_DATA_PERMANENT_BASE + path + MODEL_SUFFIX):
        whole_path = MODEL_DATA_PERMANENT_BASE + path + MODEL_SUFFIX
    else:
        whole_path = MODEL_DATA_BASE + path + MODEL_SUFFIX
    checkpoint = torch.load(whole_path)
    param = checkpoint['param']
    model = Net(param.get_net_input_size(), param.get_net_layers()).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    batch_size = checkpoint['batch_size']

    try:
        print("load model from", path, "industry\033[33m", param.industry_list,
              "\033[0m constituent list\033[33m", param.constituent_list,
              "\033[0m total iterations", checkpoint['epoch'])
        print("model predict days", param.predict_days,
              "threshold", param.predict_thresholds,
              "predict type", param.predict_types)
        val_accuracy = checkpoint['val_accuracy']
        val_precision = checkpoint['val_precision']
        val_recall = checkpoint['val_recall']
        val_f1 = checkpoint['val_f1']
        print("with loss [{:.2f}],"
              " val accuracy [{:.2f}%], val precision [{:.2f}%], val recall [{:.2f}%], val_f1 [{:.2f}%]"
              .format(loss.item(), val_accuracy, val_precision, val_recall, val_f1))
    except KeyError:
        print("failed to get validation checkpoint")
    except AttributeError:
        print("failed to get param")

    return model, optimizer, epoch, loss, batch_size, param


def get_prediction_from_param(param):
    predict_days, thresholds, predict_type = param.get_predict_param()
    assert len(predict_days) == len(thresholds)
    ret = "PREDICTIONS: stock "

    for i in range(len(predict_days)):
        threshold = str(thresholds[i] * 100) + "%"
        day = str(predict_days[i])
        if thresholds[i] >= 0 and predict_type[i] == "max":
            ret = ret + "raise to highest [\033[31m" + threshold + "\033[0m] in [\033[31m" + day + "\033[0m] days"
        elif thresholds[i] < 0 and predict_type[i] == "max":
            ret = ret + "drop to lowest [\033[32m" + threshold + "\033[0m] in [\033[32m" + day + "\033[0m] days"
        elif thresholds[i] >= 0 and predict_type[i] == "average":
            ret = ret + "avg price will be [\033[31m" + threshold + "\033[0m]" \
                                                                    " higher in [\033[31m" + day + "\033[0m] days"
        elif thresholds[i] < 0 and predict_type[i] == "average":
            ret = ret + "avg price will be [\033[32m" + threshold + "\033[0m]" \
                                                                    " lower in [\033[32m" + day + "\033[0m] days"

        if i != len(predict_days) - 1:
            ret = ret + ", "
    return ret
