import torch

from dataPool.analysisData import *
from linearRegression.linearRegressionModel import LinearRegressionModel


def calc_linear():
    # x_data = get_x()
    x_data = torch.Tensor([[3.0, 2.0], [6.0, 5.0], [9.0, 7.0], [1, 3], [3, 10],[10,3]])
    print(x_data)
    print(x_data.size())
    # x1_data = get_x()
    # print(x1_data.size())
    # y_data = get_y()
    y_data = torch.Tensor([[3.2], [6.5], [9.7], [1.3], [4.0], [10.3]])
    print(y_data)
    our_model = LinearRegressionModel()
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(our_model.parameters(), lr=0.001)
    for epoch in range(500):
        # Forward pass: Compute predicted y by passing
        # x to the model
        pred_y = our_model(x_data)

        # Compute and print loss
        loss = criterion(pred_y, y_data)

        # Zero gradients, perform a backward pass,
        # and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    new_var = torch.tensor([[2.0, 1.0]])
    # new_var = torch.rand(1, 2)
    print(new_var)
    print(new_var.size())
    pred_y = our_model(new_var)
    print(pred_y)
    print("predict (after training)",  our_model(new_var).item())







