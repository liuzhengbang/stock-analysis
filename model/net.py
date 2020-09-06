import numpy as np
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_param):
        """
        In the constructor we instantiate one nn.Linear modules and assign them as
        member variables.
        """
        super(NeuralNetwork, self).__init__()

        self.layer_para = layer_param
        print("module input size:", input_size)
        self.linear1 = nn.Linear(input_size, layer_param[0])
        self.linear2 = nn.Linear(layer_param[0], layer_param[1])
        self.linear3 = nn.Linear(layer_param[1], 1)
        self.tanh = nn.Tanh()
        self.sm = nn.Sigmoid()
        # self.sm = nn.LeakyReLU()

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
        pred = self.tanh(pred)
        # pred = self.sm(pred)
        # print(self.linear1.weight.sum())
        return pred


class LSTMNeuralNetwork(nn.Module):
    def __init__(self, input_size):

        super(LSTMNeuralNetwork, self).__init__()
        self.rnn = torch.nn.LSTM(input_size, 300, 2, batch_first=True)
        self.linear1 = torch.nn.Linear(300, 30)
        self.linear2 = torch.nn.Linear(30, 100)
        self.linear3 = torch.nn.Linear(100, 1)
        self.tanh = torch.nn.Tanh()
        self.sm = torch.nn.Sigmoid()
        self.hidden_cell = (torch.zeros(1, 1, 100),
                            torch.zeros(1, 1, 100))

    def forward(self, x):

        pred, hn = self.rnn(x)
        pred = self.linear1(pred)
        pred = self.tanh(pred)
        pred = self.linear2(pred)
        pred = self.tanh(pred)
        pred = self.linear3(pred)
        pred = self.sm(pred)
        return pred

