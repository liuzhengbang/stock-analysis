import numpy as np
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(input_size, num_classes)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.05, print_cost=False):
    # print(x_train.shape)
    logistic_model = LogisticRegression(x_train.shape[1], 1)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(logistic_model.parameters(), lr=learning_rate)

    # Training the Model
    for epoch in range(num_iterations):
        dataset_train_x = x_train

        # print("x", dataset_train_x.shape)
        dataset_train_y = y_train
        # print("y", dataset_train_y.shape)
        # Forward + Backward + Optimize

        train_predict_out = logistic_model(dataset_train_x)
        cost = criterion(train_predict_out, dataset_train_y)
        # print_loss = cost.data.item()
        # mask = train_predict_out.ge(0.5).float()
        # correct = (mask == dataset_train_y).sum()
        # acc = correct.item() / dataset_train_x.size(0)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if print_cost and epoch % 100 == 0:
            print("Cost after iteration %i: %f" % (epoch, cost))
            # print('acc is {:.4f}'.format(acc))

    y_prediction_test = predict(logistic_model, x_test)
    y_prediction_train = predict(logistic_model, x_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train.numpy())) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test.numpy())) * 100))

    return logistic_model


def predict(module, source):
    m = source.shape[0]
    y_prediction = np.zeros((m, 1))
    red = module.forward(source)
    for i in range(red.shape[0]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if red[i, 0] <= 0.5:
            y_prediction[i, 0] = 0
        else:
            y_prediction[i, 0] = 1

    assert (y_prediction.shape == (m, 1))

    return y_prediction
