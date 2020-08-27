import torch
import torch.nn as nn

from data_provider.data_constructor import  construct_dataset, DataException
from net.model import save, load
from net.net import NeuralNetwork as Net
from torch.utils.data.dataset import Dataset
device = torch.device('cuda:0')


class TrainingDataset(Dataset):
    def __init__(self, positive_data, negative_data):
        self.pos_data = positive_data
        self.neg_data = negative_data
        self.pos_length = len(self.pos_data)
        self.neg_length = len(self.neg_data)
        self.length = max(self.pos_length, self.neg_length) * 2

    def __getitem__(self, ndx):
        index = ndx // 2
        # print("index", index, "in", ndx)
        if ndx % 2 == 0:
            pos_index = index % self.pos_length
            # print("pos index", pos_index)
            return self.pos_data.values[pos_index], torch.tensor([1.0])
        else:
            neg_index = index % self.neg_length
            # print("neg index", neg_index)
            return self.neg_data.values[neg_index], torch.tensor([0.0])

    def __len__(self):
        # print("length", self.length)
        return self.length


def train_model(loader, x_test, y_test, param, prev_model=None, num_iterations=2000, learning_rate=0.9, weight=1,
                print_cost=False):
    print("start training")

    if prev_model is not None:
        model, optimizer, epoch_prev, loss, param_prev = load(prev_model)
        param.set_x_size(param_prev.get_x_size())
        param.set_net_param(param_prev.get_net_param())
    else:
        input_size = x_test.shape[1]
        net_param = [2000, 100]
        model = Net(input_size, net_param).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        epoch_prev = 0
        loss = -1
        param.set_x_size(input_size)
        param.set_net_param(net_param)

    pos_weight = torch.tensor([weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=pos_weight)
    epoch = 0

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

        if print_cost and epoch % 10 == 0:
            grad_sum = torch.tensor(0.0)
            for p in model.parameters():
                grad_sum += p.grad.norm()
            with torch.no_grad():
                accuracy, precision, recall = validate(model, x_test, y_test)
            print("Loss after iteration {} with loss: {:.6f}, grad sum: {:.6f},"
                  " test accuracy {}%, precision {}%, recall {}%"
                  .format(epoch, loss.data, grad_sum.data, accuracy, precision, recall))

    accuracy, precision, recall = validate(model, x_test, y_test)
    print("Test Dataset Accuracy:", accuracy, "Precision:", precision, "Recall:", recall)
    save(model, param, epoch + epoch_prev + 1, optimizer, loss, accuracy, precision, recall)

    return model


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
    m = source.shape[0]
    y_prediction = torch.zeros((m, 1), device=device)
    prob = -1
    ret = module.forward(source)
    for i in range(ret.shape[0]):
        prob = round(ret[i, 0].item(), 5)

        # Convert probabilities A[0,i] to actual predictions p[0,i] For Sigmoid
        if ret[i, 0] <= 0.0:
            y_prediction[i, 0] = 0
        else:
            y_prediction[i, 0] = 1

    assert (y_prediction.shape == (m, 1))

    return y_prediction.item(), prob


def validate(module, source, y_test):
    y_prediction = predict(module, source)
    total_sample = source.shape[0]
    accuracy = 100.0 - torch.mean(torch.abs(torch.sub(y_prediction, y_test))) * 100.0
    total_positive_prediction = 0
    total_negative_prediction = 0
    true_positive = 0
    false_negative = 0
    all_positive = torch.sum(y_prediction)
    for i in range(total_sample):
        if y_prediction[i, 0] == 0.0:
            total_negative_prediction = total_negative_prediction + 1
            if y_test[i, 0] == 0.0:
                false_negative = false_negative + 1
        elif y_prediction[i, 0] == 1.0:
            total_positive_prediction = total_positive_prediction + 1
            if y_test[i, 0] == 1.0:
                true_positive = true_positive + 1
    assert total_sample == (total_positive_prediction + total_negative_prediction)

    precision = 100.0 * true_positive / all_positive
    if (true_positive + false_negative) == 0:
        recall = 0.00
    else:
        recall = 100.0 * true_positive / (true_positive + false_negative)
    accuracy = round(accuracy.item(), 2)
    precision = round(precision.item(), 2)
    recall = round(recall, 2)

    return accuracy, precision, recall


def validate_model(model_name, stock_list, index_list_analysis, predict_days, thresholds, predict_type):
    model, _, _, _, _ = load(model_name)
    for stock in stock_list:
        try:
            x_test, y_test = construct_dataset(stock, index_list_analysis,
                                               predict_days=predict_days, thresholds=thresholds, predict_type=predict_type,
                                               return_data=True)
        except DataException:
            continue

        with torch.no_grad():
            accuracy, precision, recall = validate(model, x_test, y_test)
        print("stock {}: total sample {}, positive sample {}, accuracy {}%, precision {}%, recall {}%"
              .format(stock, len(x_test), sum(y_test).item(), accuracy, precision, recall))



