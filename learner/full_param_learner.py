import torch.nn as nn
import torch.nn.functional as F

from GSL.learner import BaseLearner


class FullParam(BaseLearner):
    """Full graph parameterization learner"""

    def __init__(
        self,
        metric=None,
        processors=None,
        features=None,
        sparse=False,
        non_linearize=None,
        adj=None,
    ):
        super(FullParam, self).__init__(metric, processors)
        self.sparse = sparse
        self.non_linearize = non_linearize
        if adj is None:
            adj = self.metric(features)
        if processors:
            for processor in self.processors:
                adj = processor(adj)
        self.adj = nn.Parameter(adj)

    def forward(self, h):
        # note: non_linearize should be conducted in every forward
        adj = self.non_linearize(self.adj)
        return adj
