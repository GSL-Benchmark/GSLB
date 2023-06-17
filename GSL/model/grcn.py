from GSL.model import BaseModel
from GSL.encoder.gcn import GCNConv, GCNConv_diag
from GSL.metric import InnerProductSimilarity
from GSL.utils import accuracy

import torch
import torch.nn.functional as F
from tqdm import tqdm

EOS = 1e-10


class GRCN(BaseModel):
    """Graph-Revised Convulutional Network (ECML-PKDD 2020)"""
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device):
        super(GRCN, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device)
        self.num_features = num_features
        self.graph_nhid = int(self.config.hid_graph.split(":")[0])
        self.graph_nhid2 = int(self.config.hid_graph.split(":")[1])
        self.nhid = self.config.nhid
        self.conv1 = GCNConv(num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, num_classes)

        if self.config.sparse:
            self.conv_graph = GCNConv_diag(num_features)
            self.conv_graph2 = GCNConv_diag(num_features)
        else:
            self.conv_graph = GCNConv(num_features, self.graph_nhid)
            self.conv_graph2 = GCNConv(self.graph_nhid, self.graph_nhid2)

        self.F = ({'relu': F.relu, 'prelu': F.prelu, 'tanh': torch.tanh})[self.config.F]
        self.F_graph = ({'relu': F.relu, 'prelu': F.prelu, 'tanh': torch.tanh})[self.config.F_graph]
        self.dropout = self.config.dropout
        self.K = self.config.compl_param.split(":")[0]
        self.mask = None
        self.Adj_new = None
        self._normalize = self.config.normalize
        self.reduce = self.config.reduce
        self.sparse = self.config.sparse
        self.norm_mode = "sym"
        self.config = self.config

    def init_para(self):
        self.conv1.init_para()
        self.conv2.init_para()
        self.conv_graph.init_para()
        self.conv_graph2.init_para()

    def graph_parameters(self):
        return list(self.conv_graph.parameters()) + list(self.conv_graph2.parameters())

    def base_parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    def normalize(self, adj, mode="sym"):
        if not self.sparse:
            if mode == "sym":
                inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
                return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
            elif mode == "row":
                inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
                return inv_degree[:, None] * adj
            else:
                exit("wrong norm mode")
        else:
            adj = adj.coalesce()
            if mode == "sym":
                inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + EOS)
                D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
            elif mode == "row":
                inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
                D_value = inv_degree[adj.indices()[0]]
            else:
                exit("wrong norm mode")
            new_values = adj.values() * D_value
            return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).to(self.device)

    def _node_embeddings(self, features, adj):
        norm_adj = self.normalize(adj, self.norm_mode)
        node_embeddings = self.F_graph(self.conv_graph(features, norm_adj, self.sparse))
        node_embeddings = self.conv_graph2(node_embeddings, norm_adj, self.sparse)
        if self._normalize:
            node_embeddings = F.normalize(node_embeddings, dim=1, p=2)
        return node_embeddings

    def _sparse_graph(self, raw_graph, K):
        if self.reduce == 'knn':
            values, indices = raw_graph.topk(k=int(K), dim=-1)
            assert torch.sum(torch.isnan(values)) == 0
            assert torch.max(indices) < raw_graph.shape[1]
            if not self.sparse:
                self.mask = torch.zeros(raw_graph.shape).to(self.device)
                self.mask[torch.arange(raw_graph.shape[0]).view(-1,1), indices] = 1.
                self.mask[indices, torch.arange(raw_graph.shape[1]).view(-1,1)] = 1.
            else:
                inds = torch.stack([torch.arange(raw_graph.shape[0]).view(-1,1).expand(-1,int(K)).contiguous().view(1,-1)[0].to(self.device),
                                     indices.view(1,-1)[0]])
                inds = torch.cat([inds, torch.stack([inds[1], inds[0]])], dim=1)
                values = torch.cat([values.view(1,-1)[0], values.view(1,-1)[0]])
                return inds, values
        else:
            exit("wrong sparsification method")
        self.mask.requires_grad = False
        sparse_graph = raw_graph * self.mask
        return sparse_graph

    def feedforward(self, features, adj):
        adj.requires_grad = False

        node_embeddings = self._node_embeddings(features, adj)

        metric = InnerProductSimilarity()
        adj_new = metric(node_embeddings[:, :int(self.num_features/2)], node_embeddings[:, :int(self.num_features/2)])
        adj_new += metric(node_embeddings[:, int(self.num_features/2):], node_embeddings[:, int(self.num_features/2):])

        if not self.sparse:
            adj_new = self._sparse_graph(adj_new, self.K)
            adj_new = self.normalize(adj + adj_new, self.norm_mode)
        else:
            adj_new_indices, adj_new_values = self._sparse_graph(adj_new, self.K)
            new_inds = torch.cat([adj.coalesce().indices(), adj_new_indices], dim=1)
            new_values = torch.cat([adj.coalesce().values(), adj_new_values])
            adj_new = torch.sparse.FloatTensor(new_inds, new_values, adj.size()).to(self.device)
            adj_new = self.normalize(adj_new, self.norm_mode)

        x = self.conv1(features, adj_new, self.sparse)
        x = F.dropout(self.F(x), training=self.training, p=self.dropout)
        x = self.conv2(x, adj_new, self.sparse)

        return F.log_softmax(x, dim=1)

    def test(self, features, adj, labels, mask):
        with torch.no_grad():
            output = self.feedforward(features, adj)
        result = self.metric(output[mask], labels[mask])
        return result.item()

    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        if adj.is_sparse:
            indices = adj.coalesce().indices()
            values = adj.coalesce().values()
            shape = adj.coalesce().shape
            num_nodes = features.shape[0]
            loop_edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(adj.device)
            loop_edge_index = torch.cat([indices, loop_edge_index], dim=1)
            loop_values = torch.ones(num_nodes).to(adj.device)
            loop_values = torch.cat([values, loop_values], dim=0)
            adj = torch.sparse_coo_tensor(indices=loop_edge_index, values=loop_values, size=shape)
        else:
            adj += torch.eye(adj.shape[0]).to(self.device)
            adj = adj.to_sparse()

        optimizer_base = torch.optim.Adam(self.base_parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        optimizer_graph = torch.optim.Adam(self.graph_parameters(), lr=self.config.lr_graph, weight_decay=self.config.weight_decay_graph)

        best_val_result = 0
        for epoch in range(self.config.num_epochs):
            optimizer_base.zero_grad()
            optimizer_graph.zero_grad()

            output = self.feedforward(features, adj)

            loss = F.nll_loss(output[train_mask], labels[train_mask])
            loss.backward(retain_graph=False)

            optimizer_base.step()
            optimizer_graph.step()

            if epoch % 10 == 0:
                train_result = self.test(features, adj, labels, train_mask)
                val_result = self.test(features, adj, labels, val_mask)
                test_result = self.test(features, adj, labels, test_mask)
                if val_result > best_val_result:
                    best_val_result = val_result
                    self.best_result = test_result

                print(f'Epoch: {epoch: 02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_result:.2f}%, '
                      f'Valid: {100 * val_result:.2f}%, '
                      f'Test: {100 * test_result:.2f}%')

