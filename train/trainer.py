import time

import numpy as np
import torch
import torch.nn as nn
from train.net import NeuralNetwork as Net
device = torch.device('cuda:0')


def train_model(loader, x_test, y_test, num_iterations=2000, learning_rate=0.9, print_cost=False):
    print("start training")

    # print(x_train.shape)

    model = Net(x_test.shape[1]).to(device=device)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # x_train = x_train.reshape(3238, 1, 8)
    # x_train = x_train.to(device=device)
    # y_train = y_train.reshape(3238, 1, 1)
    # y_train = y_train.to(device=device)
    # x_test = x_test.reshape(2338, 1, 8)
    # x_test = x_test.to(device=device)
    # y_test = y_test.reshape(2338, 1, 1)
    # y_test = y_test.to(device=device)

    # Training the Model
    for epoch in range(num_iterations):
        for i, (x, y) in enumerate(loader):
            train_predict_out = model(x)
            loss = criterion(train_predict_out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if print_cost and epoch % 1 == 0:
            print("Loss after iteration %i: %f" % (epoch, loss))
            # print('acc is {:.4f}'.format(acc))

    y_prediction_test = predict(model, x_test)
    # y_prediction_train = predict(model, x_train)
    # print(y_prediction_train.shape, y_train.shape)
    # print("train accuracy: {} %".format(100 - torch.mean(torch.abs(torch.sub(y_prediction_train, y_train))) * 100))
    # print("test accuracy: {} %".format(100 - torch.mean(torch.abs(torch.sub(y_prediction_test, y_test))) * 100))

    # train_accuracy = 100 - torch.mean(torch.abs(torch.sub(y_prediction_train, y_train))) * 100
    # train_accuracy = round(train_accuracy.item(), 2)
    test_accuracy = 100 - torch.mean(torch.abs(torch.sub(y_prediction_test, y_test))) * 100
    test_accuracy = round(test_accuracy.item(), 2)

    # print("train accuracy", train_accuracy, "%")
    print("test accuracy", test_accuracy, "%")

    # l1_w, l1_b, l2_w, l2_b = model.weight()
    # print("l1 weight", l1_w)
    # print("l1 bias", l1_b)
    # print("l2 weight", l2_w)
    # print("l2 bias", l2_b)
    str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    torch.save(model, "model_data/" + str_time + "-" + str(test_accuracy) + "-model.pt")

    return model


def predict(module, source):
    m = source.shape[0]
    y_prediction = torch.zeros((m, 1), device=device)
    red = module.forward(source)
    for i in range(red.shape[0]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if red[i, 0] <= 0.5:
            y_prediction[i, 0] = 0
        else:
            y_prediction[i, 0] = 1

    assert (y_prediction.shape == (m, 1))

    return y_prediction
