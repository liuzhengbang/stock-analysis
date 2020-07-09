import time

import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda:0')


class Net(nn.Module):
    def __init__(self, input_size):
        """
        In the constructor we instantiate one nn.Linear modules and assign them as
        member variables.
        """
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 30)
        self.linear2 = torch.nn.Linear(30, 100)
        self.linear3 = torch.nn.Linear(100, 1)
        self.tanh = torch.nn.Tanh()
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        pred = self.linear1(x)
        pred = self.tanh(pred)
        pred = self.linear2(pred)
        pred = self.tanh(pred)
        pred = self.linear3(pred)
        pred = self.sm(pred)
        return pred

    def weight(self):
        return self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias


def train_model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.9, print_cost=False):

    # print(x_train.shape)
    model = Net(x_train.shape[1]).to(device=device)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    x_train = x_train.to(device=device)

    # print("x", dataset_train_x.shape)
    y_train = y_train.to(device=device)

    x_test = x_test.to(device=device)
    y_test = y_test.to(device=device)
    # Training the Model
    for epoch in range(num_iterations):

        # print("y", dataset_train_y.shape)
        # Forward + Backward + Optimize

        train_predict_out = model(x_train)
        loss = criterion(train_predict_out, y_train)
        # print_loss = loss.data.item()
        # mask = train_predict_out.ge(0.5).float()
        # correct = (mask == dataset_train_y).sum()
        # acc = correct.item() / dataset_train_x.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if print_cost and epoch % 100 == 0:
            print("Loss after iteration %i: %f" % (epoch, loss))
            # print('acc is {:.4f}'.format(acc))

    y_prediction_test = predict(model, x_test)
    y_prediction_train = predict(model, x_train)
    # print(y_prediction_train.shape, y_train.shape)
    # print("train accuracy: {} %".format(100 - torch.mean(torch.abs(torch.sub(y_prediction_train, y_train))) * 100))
    # print("test accuracy: {} %".format(100 - torch.mean(torch.abs(torch.sub(y_prediction_test, y_test))) * 100))

    train_accuracy = 100 - torch.mean(torch.abs(torch.sub(y_prediction_train, y_train))) * 100
    train_accuracy = round(train_accuracy.item(), 2)
    test_accuracy = 100 - torch.mean(torch.abs(torch.sub(y_prediction_test, y_test))) * 100
    test_accuracy = round(test_accuracy.item(), 2)

    print("train accuracy", train_accuracy, "%")
    print("test accuracy", test_accuracy, "%")

    l1_w, l1_b, l2_w, l2_b = model.weight()
    # print("l1 weight", l1_w)
    # print("l1 bias", l1_b)
    # print("l2 weight", l2_w)
    # print("l2 bias", l2_b)
    str_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    torch.save(model, str_time + "--" + str(train_accuracy) + "-" + str(test_accuracy) + "-model.pt")

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
