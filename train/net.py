import numpy as np
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        """
        In the constructor we instantiate one nn.Linear modules and assign them as
        member variables.
        """
        super(NeuralNetwork, self).__init__()
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
