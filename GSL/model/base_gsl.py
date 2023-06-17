from GSL.learner import *
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    '''
    Abstract base class for graph structure learning models.

    learner:
        Graph learner module.
    device: str
        'cpu' or 'cuda'
    mode: str
        'structure_inference' or 'structure_refinement'
    sparse: bool
        True or False
    '''
    def __init__(self, device):
        super(BaseModel, self).__init__()
        self.device = device

    def check_adj(self, adj):
        """
        Check if the modified adjacency is symmetric and unweighted.
        """
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.tocsr().max() == 1, "Max value should be 1!"
        assert adj.tocsr().min() == 0, "Min value should be 0!"

    