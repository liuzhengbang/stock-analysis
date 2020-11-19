import numpy as np
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_param):
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
        pred = self.linear1(x)
        pred = self.tanh(pred)
        pred = self.linear2(pred)
        pred = self.tanh(pred)
        pred = self.linear3(pred)
        pred = self.tanh(pred)
        # pred = self.sm(pred)
        # print(self.linear1.weight.sum())
        return pred


class LSTMNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):

        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        print(x.shape, out.shape)
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

