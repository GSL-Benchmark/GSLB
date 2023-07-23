import torch
import torch.nn as nn
from GSL.utils import knn_fast
import torch.nn.functional as F
import dgl

from GSL.learner import BaseLearner

class MLPLearner(BaseLearner):
    """Multi-Layer Perception learner"""
    def __init__(self, metric, processors, nlayers, in_dim, hidden_dim, activation, sparse, k=None):
        super().__init__(metric=metric, processors=processors)

        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(1, nlayers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.param_init()
        self.activation = activation
        self.sparse = sparse
        self.k = k

    def param_init(self):
        for layer in self.layers:
            # layer.reset_parameters()
            layer.weight = nn.Parameter(torch.eye(self.in_dim))

    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
        return x

    def forward(self, features):
        if self.sparse:
            z = self.internal_forward(features)
            # TODO: knn should be moved to processor
            rows, cols, values = knn_fast(z, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = self.activation(values_)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            z = self.internal_forward(features)
            z = F.normalize(z, dim=1, p=2)
            similarities = self.metric(z)
            for processor in self.processors:
                similarities = processor(similarities)
            similarities = F.relu(similarities)
            return similarities