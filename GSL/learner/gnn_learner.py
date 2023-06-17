import torch
import dgl
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from GSL.encoder import GCNConv_dgl, GCNConv
from GSL.learner import BaseLearner
from GSL.utils import knn_fast, top_k
from GSL.metric import *
from GSL.processor import *
from GSL.data import Dataset

class GNNLearner(BaseLearner):
    def __init__(self, metric, processors, adj, nlayers, num_feature, hidden,
                 activation, sparse, k, base_model='GCN', bias=False):
        super(GNNLearner, self).__init__(metric, processors)

        self.nlayers = nlayers
        if sparse:
            self.base_model = {'GCN': GCNConv_dgl}[base_model]
            self.layers = nn.ModuleList()
            if nlayers == 1:
                self.layers.append(self.base_model(num_feature, hidden))
            else:
                self.layers.append(self.base_model(num_feature, hidden))
                for _ in range(nlayers - 2):
                    self.layers.append(self.base_model(hidden, hidden))
                self.layers.append(self.base_model(hidden, hidden))
        else:
            self.base_model = {'GCN': GCNConv}[base_model]
            self.layers = nn.ModuleList()
            if nlayers == 1:
                self.layers.append(self.base_model(num_feature, hidden, bias=bias, activation=activation))
            else:
                self.layers.append(self.base_model(num_feature, hidden, bias=bias, activation=activation))
                for _ in range(nlayers - 2):
                    self.layers.append(self.base_model(hidden, hidden, bias=bias, activation=activation))
                self.layers.append(self.base_model(hidden, hidden, bias=bias, activation=activation))

        self.activation = activation
        self.dim = hidden
        self.sparse = sparse
        self.k = k
        self.adj = adj
        self.param_init()

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.dim))

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            if self.sparse:
                h = layer(self.adj, h)
            else:
                h = layer(h, self.adj)
            if i != (len(self.layers) - 1):
                h = self.activation(h)
        return h

    def forward(self, features, v_indices=None):
        # TODO: knn sparsify should be a post-processor
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = self.processors(values_)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            # embeddings = F.normalize(embeddings, dim=1, p=2)
            if v_indices is not None:
                similarities = self.metric(embeddings, v_indices)
            else:
                similarities = self.metric(embeddings)
            for processor in self.processors:
                similarities = processor(similarities)
            return similarities


if __name__ == "__main__":
    data_path = osp.join(osp.expanduser('~'), 'datasets')
    data = Dataset(root=data_path, name='cora')
    adj = data.adj

    metric = CosineSimilarity()
    processors = [NonLinearize(non_linearity='relu', alpha=1)]
    graph_learner = GNNLearner(metric, processors, adj, 2, 32, F.relu, False, 30, 'GCN')
