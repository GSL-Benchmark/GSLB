import torch
import torch.nn as nn

class BaseLearner(nn.Module):
    """Abstract base class for graph learner"""
    def __init__(self, metric, processors):
        super(BaseLearner, self).__init__()

        self.metric = metric
        self.processors = processors