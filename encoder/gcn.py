import copy
import math

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from torch.nn import Linear, ReLU, Sequential
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_

from GSL.utils import lds_normalize_adjacency_matrix
from .metamodule import MetaModule, MetaLinear, get_subdict


class GCNConv(nn.Module):
    def __init__(self, input_size, output_size, residual=False, bias=False, activation=None):
        super(GCNConv, self).__init__()

        #self.device = device

        #self.linear = nn.Linear(input_size, output_size).to(self.device)
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.residual = residual

        if bias:
            #self.bias = Parameter(torch.FloatTensor(output_size).to(self.device))
            self.bias = Parameter(torch.FloatTensor(output_size))
        else:
            self.register_parameter('bias', None)

        self.init_para()

    def init_para(self):
        self.linear.reset_parameters()
        std = 1. / math.sqrt(self.linear.weight.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input, A, sparse=False):

        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        if self.bias is not None:
            output = output + self.bias
        if self.residual:
            output = input + output
        if self.activation is not None:
            output = self.activation(output)
        return output


class GCNConv_diag(torch.nn.Module):
    '''
    A GCN convolution layer of diagonal matrix multiplication
    '''
    def __init__(self, input_size, device):
        super(GCNConv_diag, self).__init__()
        self.device = device

        self.W = torch.nn.Parameter(torch.ones(input_size).to(self.device))
        self.input_size = input_size

    def init_para(self):
        self.W = torch.nn.Parameter(torch.ones(self.input_size).to(self.device))

    def forward(self, input, A, sparse=False):
        hidden = input @ torch.diag(self.W)
        # hidden = torch.sparse.mm(self.mW, input.t()).t()
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output


class GCNConv_dgl(nn.Module):
    """
    This graph conv layer will be deprecated, which was once used by SLAPS.
    """
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['feat'] = self.linear(x)
            g.update_all(fn.u_mul_e('feat', 'w', 'm'), fn.sum(msg='m', out='feat'))
            return g.ndata['feat']


class SparseDropout(nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - dprob

    def forward(self, x):
        mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(torch.bool)
        rc = x._indices()[:,mask]
        val = x._values()[mask]*(1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, device, sparse=False, residual=False, activation_last=None, conv_bias=False, bn=False):
        super(GCN, self).__init__()

        self.residual = residual
        self.layers = nn.ModuleList()
        self.device = device
        self.bn = bn
        if self.bn:
            self.bn_list = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bn_list.append(nn.BatchNorm1d(hidden_channels))

        if self.residual:
            self.in_linear = nn.Linear(in_channels, hidden_channels)
            self.out_linear = nn.Linear(hidden_channels, out_channels)
            for _ in range(num_layers):
                self.layers.append(GCNConv(hidden_channels, hidden_channels, bias=conv_bias, residual=residual))
        else:
            self.layers.append(GCNConv(in_channels, hidden_channels, bias=conv_bias))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels, bias=conv_bias))
            self.layers.append(GCNConv(hidden_channels, out_channels, bias=conv_bias))

        self.dropout = dropout
        if not sparse:
            self.dropout_adj = nn.Dropout(p=dropout_adj)
        else:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        self.sparse = sparse
        self.activation_last = activation_last

    def forward(self, x, adj_t, return_hidden=False):
        # from IPython import embed; embed(header='gcn.py, line 131')
        Adj = self.dropout_adj(adj_t)
        #x = x.to(self.device)
        is_torch_sparse_tensor = Adj.is_sparse
        outputs = []

        if self.residual:
            x = self.in_linear(x)
            for conv in self.layers:
                x = conv(x, Adj, sparse=is_torch_sparse_tensor)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                outputs.append(x)
            x = self.out_linear(x)
        elif self.bn:
            for _, conv in enumerate(self.layers[:-1]):
                x = conv(x, Adj, sparse=is_torch_sparse_tensor)
                x = self.bn_list[_](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                outputs.append(x)
            x = self.layers[-1](x, Adj, sparse=is_torch_sparse_tensor)
        else:
            for _, conv in enumerate(self.layers[:-1]):
                x = conv(x, Adj, sparse=is_torch_sparse_tensor)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                outputs.append(x)
            x = self.layers[-1](x, Adj, sparse=is_torch_sparse_tensor)

        if self.activation_last == 'log_softmax':
            x = F.log_softmax(x, dim=1)
        outputs.append(x)
        if return_hidden:
            return tuple(outputs)
        return x


    def reset_parameters(self):
        for conv in self.layers:
            conv.init_para()




class GCN_dgl(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, activation_last=None, bn=False, allow_zero_in_degree=False):
        super(GCN_dgl, self).__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        if self.bn:
            self.bn_list = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bn_list.append(nn.BatchNorm1d(hidden_channels))

        self.layers.append(GraphConv(in_channels, hidden_channels, allow_zero_in_degree=allow_zero_in_degree))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_channels, hidden_channels, allow_zero_in_degree=allow_zero_in_degree))
        self.layers.append(GraphConv(hidden_channels, out_channels, allow_zero_in_degree=allow_zero_in_degree))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.activation_last = activation_last

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def _stochastic_forward(self, blocks, x):

        for i, (conv, block) in enumerate(zip(self.layers[:-1], blocks[:-1])):
            x = conv(block, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layers[-1](blocks[-1], x)
        if self.activation_last == 'log_softmax':
            x = F.log_softmax(x, dim=1)
        if self.activation_last == 'relu':
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, adj_t, stochastic=False):
        Adj = adj_t
        if stochastic:
            return self._stochastic_forward(Adj, x)

        if self.dropout_adj_p > 0:
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(Adj, x)
            if self.bn: x = self.bn_list[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](Adj, x)
        if self.activation_last == 'log_softmax':
            x = F.log_softmax(x, dim=1)
        if self.activation_last == 'relu':
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv(hidden_dim, emb_dim))

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))

    def forward(self,x, Adj_, branch=None):

        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        z = self.proj_head(x)
        return z, x


class MetaDenseGraphConvolution(MetaModule):
    def __init__(self, in_features, out_features, use_bias=True):
        super(MetaDenseGraphConvolution, self).__init__()
        self.fc = MetaLinear(in_features, out_features, bias=use_bias)
        self.reset_weights()

    def reset_weights(self):
        self.fc.weight = Parameter(xavier_uniform_(self.fc.weight.clone()))
        self.fc.bias = Parameter(self.fc.bias.clone().zero_())

    def forward(self, node_features, dense_adj, params=None):
        embeddings = self.fc.forward(node_features, params=get_subdict(params, "fc"))
        return torch.mm(dense_adj, embeddings)


class MetaDenseGCN(MetaModule):
    """
    Only for LDS-GNN.
    """
    def __init__(self, in_features, hidden_features, out_features, dropout, normalize_adj: bool = True):
        super(MetaDenseGCN, self).__init__()
        self.layer_in = MetaDenseGraphConvolution(in_features, hidden_features)
        self.layer_out = MetaDenseGraphConvolution(hidden_features, out_features)

        self.dropout = dropout
        self.normalize_adj = normalize_adj

    def reset_weights(self):
        self.layer_in.reset_weights()
        self.layer_out.reset_weights()

    def forward_to_last_layer(self, node_features, dense_adj, params=None):
        # from IPython import embed; embed(header='in forward_to_last_layer')
        if self.normalize_adj:
            dense_adj = lds_normalize_adjacency_matrix(dense_adj)

        embeddings = F.dropout(node_features, self.dropout, training=self.training)
        embeddings = F.relu(self.layer_in(embeddings, dense_adj, params=get_subdict(params, 'layer_in')))
        embeddings = F.dropout(embeddings, self.dropout, training=self.training)
        return self.layer_out(embeddings, dense_adj, params=get_subdict(params, 'layer_out'))

    def forward(self, node_features, dense_adj, params=None):
        embeddings = self.forward_to_last_layer(node_features, dense_adj, params=params)
        # from IPython import embed; embed(header='in forward')
        return F.log_softmax(embeddings, dim=1)