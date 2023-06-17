import torch
import torch.nn as nn
import dgl

from GSL.learner import BaseLearner
from GSL.metric import *
from GSL.processor import *
from GSL.utils import knn_fast


class Attentive(nn.Module):
    def __init__(self, size):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(size))

    def forward(self, x):
        return x @ torch.diag(self.w)


class AttLearner(BaseLearner):
    """Attentive Learner"""
    def __init__(self, metric, processors, nlayers, size, activation, sparse):
        """
        nlayers: int
            Number of attention layers
        act: str
            Activation Function
        """
        super(AttLearner, self).__init__(metric, processors)
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        for _ in range(self.nlayers):
            self.layers.append(Attentive(size))
        self.activation = activation
        self.sparse = sparse

    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != (self.nlayers - 1):
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


if __name__ == '__main__':
    metric = CosineSimilarity()
    processors = [KNNSparsify(3), NonLinearize(non_linearity='relu'), Symmetrize(), Normalize(mode='sym')]
    activation = nn.ReLU
    att_learner = AttLearner(metric, processors, 1, 16, activation)
    x = torch.rand(8, 16)
    adj = att_learner(x)
    print(adj)