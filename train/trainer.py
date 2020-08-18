import time

import numpy as np
import torch
import torch.nn as nn

from data_provider.data_constructor import convert_to_tensor
from train.net import NeuralNetwork as Net

device = torch.device('cuda:0')


def train_model(loader, x_test, y_test, num_iterations=2000, learning_rate=0.9, weight=1, print_cost=False):
    print("start training")

    model = Net(x_test.shape[1]).to(device=device)

    pos_weight = torch.tensor([weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=pos_weight)
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
                accuracy, precision, recall = validate(model, x_test, y_test)
            print("Loss after iteration {} with loss: {:.6f}, grad sum: {:.6f},"
                  " test accuracy {}%, precision {}%, recall {}%"
                  .format(epoch, loss.data, grad_sum.data, accuracy, precision, recall))

    accuracy, precision, recall = validate(model, x_test, y_test)
    print("Test Dataset Accuracy:", accuracy, "Precision:", precision, "Recall:", recall)

    str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    torch.save(model.state_dict(),
               "model_data/" + str_time + "-" + str(accuracy) + "-" + str(precision) + "-" + str(recall) + "-model.pt")

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
