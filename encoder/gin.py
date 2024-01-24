import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        h = F.relu(self.linears[0](x))
        h = F.relu(self.linears[1](h))
        h = self.batch_norm(h)
        return h

    def reset_parameters(self):
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
        self.batch_norm.reset_parameters()


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            elif layer == num_layers - 1:
                mlp = MLP(hidden_dim, hidden_dim, output_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=True)
            )
            # self.batch_norms = nn.ModuleList()
            # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def reset_parameters(self):
        for layer in self.ginlayers:
            layer.apply_func.reset_parameters()
            layer.eps.data.fill_(0.)

    def forward(self, x, adj):
        for layer in self.ginlayers:
            x = layer(adj, x)
            # h = self.batch_norms[i](h)
            # x = F.relu(x)
        return x
