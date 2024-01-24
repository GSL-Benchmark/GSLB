import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_layer, input_dim, output_dim, n_hidden, dropout=0.5, activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(input_dim, n_hidden))
        # hidden layers
        for i in range(n_layer - 2):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, output_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        return

    def forward(self, input):
        h = input
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            # h = F.relu(layer(h))
            h = self.activation(layer(h))
        return h