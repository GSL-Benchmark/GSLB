import torch
import torch.nn as nn
from GSL.utils import knn_fast
import torch.nn.functional as F
import dgl
from GSL.learner import BaseLearner


class HadamardProduct(nn.Module):
    def __init__(self, dim):
        super(HadamardProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        return h * self.weight


class KHeadHPLearner(BaseLearner):
    def __init__(self, metric, processors, dim, num_heads, sparse=False):
        super().__init__(metric=metric, processors=processors)

        self.dim = dim
        self.num_heads = num_heads
        self.sparse = sparse

        self.hp_heads = nn.ModuleList()
        for i in range(num_heads):
            self.hp_heads.append(HadamardProduct(dim))

    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
        return x

    def forward(self, left_features, right_feature=None):
        """
        Note: The right_feature is used to computer a non-square adjacent matrix, such as relation subgraph
              of heterogeneous graph.
        """
        if self.sparse:
            raise NameError('We dont support the sparse version yet')
        else:
            if torch.sum(left_features) == 0 or torch.sum(right_feature) == 0:
                return torch.zeros((left_features.shape[0], right_feature.shape[0]))
            s = torch.zeros((left_features.shape[0], right_feature.shape[0])).to(left_features.device)
            zero_lines = torch.nonzero(torch.sum(left_features, 1) == 0)
            # The ReLU function will generate zero lines, which lead to the nan (devided by zero) problem.
            if len(zero_lines) > 0:
                left_features[zero_lines, :] += 1e-8

            # metric
            for i in range(self.num_heads):
                weighted_left_h = self.hp_heads[i](left_features)
                weighted_right_h = self.hp_heads[i](right_feature)
                s += self.metric(weighted_left_h, weighted_right_h)
            s /= self.num_heads

            # processor
            for processor in self.processors:
                s = processor(s)

            return s