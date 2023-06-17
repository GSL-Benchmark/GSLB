from GSL.learner import BaseLearner
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8


class SpLearner(BaseLearner):
    """Sparsification learner"""
    def __init__(self, nlayers, in_dim, hidden, activation, k, weight=True, metric=None, processors=None):
        super().__init__(metric=metric, processors=processors)

        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden))
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, 1))

        self.param_init()
        self.activation = activation
        self.k = k
        self.weight = weight

    def param_init(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
        return x

    def gumbel_softmax_sample(self, logits, temperature, adj, training):
        """Draw a sample from the Gumbel-Softmax distribution"""
        r = self.sample_gumble(logits.coalesce().values().shape)
        if training is not None:
            values = torch.log(logits.coalesce().values()) + r.to(logits.device)
        else:
            values = torch.log(logits.coalesce().values())
        values /= temperature
        y = torch.sparse_coo_tensor(indices=adj.coalesce().indices(), values=values, size=adj.shape)
        return torch.sparse.softmax(y, dim=1)

    def sample_gumble(self, shape):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, features, adj, temperature, training=None):
        indices = adj.coalesce().indices().t()
        values = adj.coalesce().values()

        f1_features = torch.index_select(features, 0, indices[:, 0])
        f2_features = torch.index_select(features, 0, indices[:, 1])
        auv = torch.unsqueeze(values, -1)
        temp = torch.cat([f1_features, f2_features, auv], -1)

        temp = self.internal_forward(temp)
        z = torch.reshape(temp, [-1])

        z_matrix = torch.sparse_coo_tensor(indices=indices.t(), values=z, size=adj.shape)
        pi = torch.sparse.softmax(z_matrix, dim=1)

        y = self.gumbel_softmax_sample(pi, temperature, adj, training)
        y_dense = y.to_dense()

        top_k_v, top_k_i = torch.topk(y_dense, self.k)
        kth = torch.min(top_k_v, -1)[0] + eps
        kth = kth.unsqueeze(-1)
        kth = torch.tile(kth, [1, kth.shape[0]])
        mask2 = y_dense >= kth
        mask2 = mask2.to(torch.float32)
        row_sum = mask2.sum(-1)

        dense_support = mask2

        if self.weight:
            dense_support = torch.mul(y_dense, mask2)
        else:
            print('no gradient bug here!')
            exit()

        # norm
        dense_support = dense_support + torch.eye(adj.shape[0]).to(dense_support.device)

        rowsum = torch.sum(dense_support, -1) + 1e-6
        d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
        d_mat_inv_sqer = torch.diag(d_inv_sqrt)
        ad = torch.matmul(dense_support, d_mat_inv_sqer)
        adt = ad.t()
        dadt = torch.matmul(adt, d_mat_inv_sqer)
        support = dadt

        return support