from GSL.model import BaseModel
from GSL.processor import AddEye
from GSL.encoder import NodeFormerConv
from GSL.utils import dense_adj_to_edge_index

import torch
import torch.nn.functional as F
import torch.nn as nn


class NodeFormer(BaseModel):
    """NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classiﬁcation (NeurIPS 2022)"""
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, params):
        super(NodeFormer, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device, params)

        # hyper-parameters of model initialization
        self.hidden_dim = self.config.hidden_dim
        self.num_layers = self.config.num_layers
        self.use_jk = self.config.use_jk
        self.dropout = self.config.dropout
        self.activation = F.elu
        self.use_bn = self.config.use_bn
        self.use_residual = self.config.use_residual
        self.use_act = self.config.use_act
        self.use_edge_loss = self.config.use_edge_loss
        self.num_heads = self.config.num_heads
        self.kernel_transformation = self.config.kernel_transformation
        self.nb_random_features = self.config.nb_random_features
        self.use_gumbel = self.config.use_gumbel
        self.nb_gumbel_sample = self.config.nb_gumbel_sample
        self.rb_trans = self.config.rb_trans
        
        # hyper-parameters of model training
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.weight_decay = self.config.weight_decay
        self.eval_step = self.config.eval_step
        self.lamda = self.config.lamda
        self.rb_order = self.config.rb_order
        self.tau = self.config.tau

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(num_features, self.hidden_dim))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(self.hidden_dim))
        for i in range(self.num_layers):
            self.convs.append(NodeFormerConv(self.hidden_dim, self.hidden_dim, num_heads=self.num_heads,
                                             kernel_transformation=self.kernel_transformation,
                                             nb_random_features=self.nb_random_features,
                                             use_gumbel=self.use_gumbel,
                                             nb_gumbel_sample=self.nb_gumbel_sample,
                                             rb_order=self.rb_order,
                                             rb_trans=self.rb_trans,
                                             use_edge_loss=self.use_edge_loss))
            self.bns.append(nn.LayerNorm(self.hidden_dim))

        if self.use_jk:
            self.fcs.append(nn.Linear(self.hidden_dim * self.num_layers + self.hidden_dim, num_classes))
        else:
            self.fcs.append(nn.Linear(self.hidden_dim, num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def model_parameters(self):
        params = list()
        params += list(self.convs.parameters())
        params += list(self.bns.parameters())
        params += list(self.fcs.parameters())
        return params
    
    def feedforward(self, x, adjs, tau=1.0):
        x = x.unsqueeze(0) # [B, N, H, D], B=1 denotes number of graph
        layer_ = []
        link_loss_ = []
        z = self.fcs[0](x)
        if self.use_bn:
            z = self.bns[0](z)
        z = self.activation(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        layer_.append(z)

        for i, conv in enumerate(self.convs):
            if self.use_edge_loss:
                z, link_loss = conv(z, adjs, tau)
                link_loss_.append(link_loss)
            else:
                z = conv(z, adjs, tau)
            if self.use_residual:
                z += layer_[i]
            if self.use_bn:
                z = self.bns[i+1](z)
            if self.use_act:
                z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)

        if self.use_jk: # use jk connection for each layer
            z = torch.cat(layer_, dim=-1)

        x_out = self.fcs[-1](z).squeeze(0)

        if self.use_edge_loss:
            return x_out, link_loss_
        else:
            return x_out
        
    def test(self, features, adjs, labels, mask):
        self.eval()
        with torch.no_grad():
            out, _ = self.feedforward(features, adjs, self.tau)
            acc = self.metric(out[mask], labels[mask], self.num_class)
        return acc
    
    @staticmethod
    def adj_mul(adj_i, adj, N):
        adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
        adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
        adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
        adj_j = adj_j.coalesce().indices()
        return adj_j
    
    def to(self, device):
        self.convs.to(device)
        self.fcs.to(device)
        self.bns.to(device)
        return self
    
    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask


        num_nodes = features.shape[0]
        adjs = []
        adj = AddEye()(adj)
        adj = dense_adj_to_edge_index(adj)
        adjs.append(adj)
        for i in range(self.rb_order - 1):
            adj = self.adj_mul(adj, adj, num_nodes)
            adjs.append(adj)

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model_parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_test, best_val = 0, float('-inf')

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()

            out, link_loss_ = self.feedforward(features, adjs, self.tau)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_mask], labels[train_mask])
            loss -= self.lamda * sum(link_loss_) / len(link_loss_)
            
            loss.backward()
            optimizer.step()

            if epoch % self.eval_step == 0:
                train_result = self.test(features, adjs, labels, train_mask)
                val_result = self.test(features, adjs, labels, val_mask)
                test_result = self.test(features, adjs, labels, test_mask)

                if val_result > best_val:
                    best_val = val_result
                    best_test = test_result

                print(f'Epoch: {epoch: 02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_result:.2f}%, '
                      f'Valid: {100 * val_result:.2f}%, '
                      f'Test: {100 * test_result:.2f}%')
        self.best_result = best_test.item()
