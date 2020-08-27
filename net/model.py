import time
import torch
from net.net import NeuralNetwork as Net
device = torch.device('cuda:0')
MODEL_DATA_BASE = "model_data/"


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


def save(model, param, epoch, optimizer, loss, accuracy, precision, recall):
    str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    path = str_time + "-" + str(accuracy) + "-" + str(precision) + "-" + str(recall) + "-model.pt"
    print("save model as:", path)

    torch.save({
        'epoch': epoch,
        'param': param,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'test accuracy': accuracy,
        'test precision': precision,
        'test recall': recall
    }, MODEL_DATA_BASE + path)


def load(path):
    checkpoint = torch.load(MODEL_DATA_BASE + path)
    param = checkpoint['param']
    model = Net(param.get_x_size(), param.get_net_param()).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['test accuracy']
    precision = checkpoint['test precision']
    recall = checkpoint['test recall']
    print("load model from", path, "type", param.dataset_type, "total iterations", checkpoint['epoch'])
    print("with loss", round(loss.item(), 2), "test accuracy", accuracy, "precision", precision, "recall", recall)

    return model, optimizer, epoch, loss, param
