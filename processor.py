import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class KNNSparsify:
    def __init__(self, k, discrete=False, self_loop=True):
        super(KNNSparsify, self).__init__()
        self.k = k
        self.discrete = discrete
        self.self_loop = self_loop

    def __call__(self, adj):
        _, indices = adj.topk(k=int(self.k+1), dim=-1)
        assert torch.max(indices) < adj.shape[1]
        mask = torch.zeros(adj.shape).to(adj.device)
        mask[torch.arange(adj.shape[0]).view(-1, 1), indices] = 1.

        mask.requires_grad = False
        if self.discrete:
            sparse_adj = mask.to(torch.float)
        else:
            sparse_adj = adj * mask
        
        if not self.self_loop:
            sparse_adj.fill_diagonal_(0)
        return sparse_adj


class ThresholdSparsify:
    def __init__(self, threshold):
        super(ThresholdSparsify, self).__init__()
        self.threshold = threshold

    def __call__(self, adj):
        return torch.where(adj < self.threshold, torch.zeros_like(adj), adj)


class ProbabilitySparsify:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def __call__(self, prob):
        prob = torch.clamp(prob, 0.01, 0.99)
        adj = RelaxedBernoulli(temperature=torch.Tensor([self.temperature]).to(prob.device),
                               probs=prob).rsample()
        eps = 0.5
        mask = (adj > eps).detach().float()
        adj = adj * mask + 0.0 * (1 - mask)
        return adj


class Discretize:
    def __init__(self):
        super(Discretize, self).__init__()

    def __call__(self, adj):
        adj[adj != 0] = 1.0
        return adj


class AddEye:
    def __init__(self):
        super(AddEye, self).__init__()

    def __call__(self, adj):
        adj += torch.eye(adj.shape[0]).to(adj.device)
        return adj


class LinearTransform:
    def __init__(self, alpha):
        super(LinearTransform, self).__init__()
        self.alpha = alpha

    def __call__(self, adj):
        adj = adj * self.alpha - self.alpha
        return adj


class NonLinearize:
    def __init__(self, non_linearity='relu', alpha=1.0):
        super(NonLinearize, self).__init__()
        self.non_linearity = non_linearity
        self.alpha = alpha

    def __call__(self, adj):
        if self.non_linearity == 'elu':
            return F.elu(adj) + 1
        elif self.non_linearity == 'relu':
            return F.relu(adj)
        elif self.non_linearity == 'none':
            return adj
        else:
            raise NameError('We dont support the non-linearity yet')


class Symmetrize:
    def __init__(self):
        super(Symmetrize, self).__init__()

    def __call__(self, adj):
        return (adj + adj.T) / 2


class Normalize:
    def __init__(self, mode='sym', eos=1e-10):
        super(Normalize, self).__init__()
        self.mode = mode
        self.EOS = eos

    def __call__(self, adj):
        if self.mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + self.EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif self.mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + self.EOS)
            return inv_degree[:, None] * adj
        elif self.mode == "row_softmax":
            return F.softmax(adj, dim=1)
        elif self.mode == "row_softmax_sparse":
            return torch.sparse.softmax(adj.to_sparse(), dim=1).to_dense()
        else:
            raise Exception('We dont support the normalization mode')