from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.data_constructor import construct_temp_csv_data, load_dataset, construct_dataset_instantly
from model.model_persistence import save, load
from model.net import NeuralNetwork as Net
from torch.utils.data.dataset import Dataset

from utils.consts import device


class TrainingDataset(Dataset):
    def __init__(self, positive_data, negative_data):
        self.val = False
        self.pos_data = positive_data
        self.neg_data = negative_data
        self.pos_length = len(self.pos_data)
        self.neg_length = len(self.neg_data)
        self.length = max(self.pos_length, self.neg_length) * 2
        print("training dataset length:", self.pos_length + self.neg_length,
              "with pos", self.pos_length, "samples, neg", self.neg_length, "samples,",
              "pos ratio", round(self.pos_length / (self.pos_length + self.neg_length) * 100, 2), "%")

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


class ValidationDataset(Dataset):
    def __init__(self, positive_data, negative_data):
        self.pos_data = positive_data
        self.neg_data = negative_data
        self.pos_length = len(self.pos_data)
        self.neg_length = len(self.neg_data)
        self.length = len(self.pos_data) + len(self.neg_data)
        print("validation dataset length:", self.length,
              "with pos", self.pos_length, "samples, neg", self.neg_length, "samples,",
              "pos ratio", round(len(self.pos_data) / self.length * 100, 2), "%")

    def __getitem__(self, ndx):
        if ndx < self.pos_length:
            return self.pos_data.values[ndx], torch.tensor([1.0])
        else:
            ndx = ndx - self.pos_length
            return self.neg_data.values[ndx], torch.tensor([0.0])

    def __len__(self):
        # print("length", self.length)
        return self.length


def continue_train(model_name, num_iterations, print_cost=True, weight=1):
    start_time = datetime.now()
    print("start continue training", start_time.strftime("%Y-%m-%d-%H-%M-%S"))

    model, optimizer, epoch_prev, loss, batch_size, param_prev = load(model_name)
    predict_days, predict_thresholds, predict_types = param_prev.get_predict_param()
    test_list = param_prev.get_test_stock_list()
    param_prev.set_val_date_list(_complete_val_date_list(param_prev.get_val_date_list()))
    x_test, y_test = construct_dataset_instantly(test_list[0],
                                                 index_code_list=param_prev.get_index_code_list(),
                                                 predict_days=predict_days,
                                                 predict_thresholds=predict_thresholds,
                                                 predict_types=predict_types)

    construct_temp_csv_data(param_prev.get_training_stock_list(),
                            index_code_list=param_prev.get_index_code_list(),
                            predict_days=predict_days,
                            thresholds=predict_thresholds,
                            predict_type=predict_types,
                            val_date_list=param_prev.get_val_date_list())

    train_pos_data, train_neg_data, val_pos_data, val_neg_data = load_dataset()
    train_dataset = TrainingDataset(train_pos_data, train_neg_data)
    val_dataset = ValidationDataset(val_pos_data, val_neg_data)
    loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=20000, num_workers=0, shuffle=False)

    _train(model, loader, val_loader, batch_size, epoch_prev, loss, num_iterations, optimizer, param_prev, print_cost,
           val_dataset, weight, x_test, y_test)

    end_time = datetime.now()
    time_delta = end_time - start_time
    print("continue training finished in", round(time_delta.seconds / 60 / 60, 3), "hours, ended at",
          end_time.strftime("%Y-%m-%d-%H-%M-%S"))

    return model


def train_model(train_dataset, val_dataset, x_test, y_test, param,
                batch_size=2000,
                num_iterations=2000, learning_rate=0.9,
                weight=1,
                print_cost=False):
    start_time = datetime.now()
    print("start training", start_time.strftime("%Y-%m-%d-%H-%M-%S"))
    loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=20000, num_workers=0, shuffle=False)

    input_size = x_test.shape[1]
    net_param = [3000, 300]
    model = Net(input_size, net_param).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_prev = 0
    loss = -1
    param.set_net_input_size(input_size)
    param.set_net_layers(net_param)

    _train(model, loader, val_loader, batch_size, epoch_prev, loss, num_iterations, optimizer, param, print_cost,
           val_dataset, weight, x_test, y_test)

    end_time = datetime.now()
    time_delta = end_time - start_time
    print("training finished in", round(time_delta.seconds / 60 / 60, 3), "hours, ended at",
          end_time.strftime("%Y-%m-%d-%H-%M-%S"))

    return model


def _train(model, loader, val_loader, batch_size, epoch_prev, loss, num_iterations, optimizer, param, print_cost,
           val_dataset, weight, x_test, y_test):
    pos_weight = torch.tensor([weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=pos_weight)
    epoch = 0
    max_precision = 20
    # Training the Model
    for epoch in range(num_iterations):
        for i, (x, y) in enumerate(loader):
            x = x.to(device).float()
            y = y.to(device).float()
            train_predict_out = model(x)
            loss = criterion(train_predict_out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if print_cost and epoch % 100 == 0:
            grad_sum = torch.tensor(0.0)
            for p in model.parameters():
                grad_sum += p.grad.norm()
            with torch.no_grad():
                model.eval()
                test_accuracy, test_precision, test_recall, test_f1 = validate(model, x_test, y_test)
                val_accuracy = 0
                val_precision = 0
                val_recall = 0
                val_f1 = 0
                if val_dataset is not None:
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(device).float()
                        y_val = y_val.to(device).float()

                        temp_val_accuracy, temp_val_precision, temp_val_recall, temp_val_f1 \
                            = validate(model, x_val, y_val)
                        val_accuracy = val_accuracy + temp_val_accuracy * len(x_val) / len(val_dataset)
                        val_precision = val_precision + temp_val_precision * len(x_val) / len(val_dataset)
                        val_recall = val_recall + temp_val_recall * len(x_val) / len(val_dataset)
                        val_f1 = val_f1 + temp_val_f1 * len(x_val) / len(val_dataset)

                print("Loss after iteration {} with loss: {:.6f}, grad sum: {:.6f},"
                      " [test] accuracy {:.2f}%, precision {:.2f}%, recall {:.2f}%"
                      " [validation] accuracy {:.2f}%, precision {:.2f}%, recall {:.2f}%, val_f1 {:.2f}%"
                      .format(epoch, loss.data, grad_sum.data,
                              test_accuracy, test_precision, test_recall,
                              val_accuracy, val_precision, val_recall, val_f1))
                if val_precision > max_precision or epoch % 1000 == 0:
                    save(model, param, epoch + epoch_prev + 1, optimizer, batch_size, loss,
                         val_accuracy, val_precision, val_recall, val_f1,
                         test_accuracy, test_precision, test_recall, test_f1)
                    if val_precision > max_precision:
                        max_precision = val_precision
    test_accuracy, test_precision, test_recall, test_f1 = validate(model, x_test, y_test)
    print("Test Dataset Accuracy:", test_accuracy, "Precision:", test_precision, "Recall:", test_recall)
    save(model, param, epoch + epoch_prev + 1, optimizer, batch_size, loss,
         val_accuracy, val_precision, val_recall, val_f1,
         test_accuracy, test_precision, test_recall, test_f1)


def predict(module, source):
    m = source.shape[0]
    y_prediction = torch.zeros((m, 1), device=device)
    ret = module.forward(source)
    for i in range(ret.shape[0]):

        # Convert probabilities A[0,i] to actual predictions p[0,i] For Sigmoid
        if ret[i, 0] <= 0.0:
            y_prediction[i, 0] = 0
        else:
            y_prediction[i, 0] = 1

    assert (y_prediction.shape == (m, 1))

    return y_prediction


def predict_with_prob(module, source):
    assert source.shape[0] == 1
    m = source.shape[0]
    y_prediction = torch.zeros((m, 1), device=device)

    ret = module.forward(source)
    prob = round(ret[0, 0].item(), 5)

    # Convert probabilities A[0,i] to actual predictions p[0,i] For Sigmoid
    if ret[0, 0] <= 0.0:
        y_prediction[0, 0] = 0
    else:
        y_prediction[0, 0] = 1

    assert (y_prediction.shape == (m, 1))

    return y_prediction.item(), prob


def validate(module, source, y_test):
    y_prediction = predict(module, source)
    total_sample = source.shape[0]
    tp = ((y_prediction == 1.0) & (y_test == 1.0)).sum().item()
    tn = ((y_prediction == 0.0) & (y_test == 0.0)).sum().item()
    fp = ((y_prediction == 1.0) & (y_test == 0.0)).sum().item()
    fn = ((y_prediction == 0.0) & (y_test == 1.0)).sum().item()
    print("tp:", tp, "fp:", fp, "tn:", tn, "fn", fn)
    assert (tp + tn + fp + fn) == total_sample

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (tp + tn + fp + fn) == 0:
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    if (recall + precision) != 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0

    return accuracy * 100, precision * 100, recall * 100, f1 * 100


def _complete_val_date_list(val_list):
    delta = timedelta(days=365 * 10)
    today = datetime.today()
    recent_date = today
    for date_str in val_list:
        val_date = datetime.strptime(date_str, "%Y-%m-%d")
        delta_temp = today - val_date
        if delta_temp < delta:
            delta = delta_temp
            recent_date = val_date

    print("most recent validation date is", recent_date)
    for i in range((today - recent_date).days):
        date = datetime.today() + timedelta(days=-i)
        val_list.append(date.strftime('%Y-%m-%d'))
        print("add", date.strftime('%Y-%m-%d'), "to validation list")

    return val_list
