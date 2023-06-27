import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from GSL.model import BaseModel
from GSL.utils import knn_fast, top_k
from GSL.encoder import GCNConv_dgl, GCNConv, GCN, GCN_dgl

def get_random_mask(features, r, scale, dataset):

    if dataset == 'ogbn-arxiv' or dataset == 'minist' or dataset == 'cifar10' or dataset == 'fashionmnist':
        probs = torch.full(features.shape, 1 / r)
        mask = torch.bernoulli(probs)
        return mask

    nones = torch.sum(features > 0.0).float()
    nzeros = features.shape[0] * features.shape[1] - nones
    pzeros = nones / nzeros / r * scale

    probs = torch.zeros(features.shape).to(features.device)
    probs[features == 0.0] = pzeros
    probs[features > 0.0] = 1 / r

    mask = torch.bernoulli(probs)

    return mask


def get_homophily(adj, labels, dgl_=False):
    if dgl_:
        src, dst = adj.edges()
    else:
        src, dst = adj.detach().nonzero().t()
    homophily_ratio = 1.0 * torch.sum((labels[src] == labels[dst])) / src.shape[0]

    return homophily_ratio


class HESGSL(BaseModel):
    """
    Homophily-Enhanced Self-Supervision for Graph Structure Learning: Insights and Directions (TNNLS 2023')
    """
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device):
        super(HESGSL, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device)
        self.nclass = num_classes
        self.hid_dim_dae = self.config.hid_dim_dae
        self.hid_dim_cla = self.config.hid_dim_cla
        self.nlayers = self.config.nlayers
        self.dropout_cla = self.config.dropout_cla
        self.dropout_adj = self.config.dropout_adj
        self.mlp_dim = self.config.mlp_dim
        self.k = self.config.k
        self.sparse = self.config.sparse
        self.lr = self.config.lr
        self.lr_cla = self.config.lr_cla
        self.weight_decay = self.config.weight_decay
        self.num_epochs = self.config.num_epochs
        self.epochs_pre = self.config.epochs_pre
        self.epochs_hom = self.config.epochs_hom
        self.dataset = self.config.dataset
        self.ratio = self.config.ratio
        self.scale = self.config.scale
        self.num_hop = self.config.num_hop
        self.alpha = self.config.alpha
        self.beta = self.config.beta
        self.patience = self.config.patience
        self.eval_step = self.config.eval_step

        self.model_dae = GCN_DAE(num_features, self.hid_dim_dae, num_features, self.nlayers, self.dropout_cla, self.dropout_adj, self.mlp_dim, self.k, self.sparse).to(device)
        if self.sparse:
            self.model_cla = GCN_dgl(num_features, self.hid_dim_cla, num_classes, self.nlayers, self.dropout_cla, self.dropout_adj)
        else:
            self.model_cla = GCN(num_features, self.hid_dim_cla, num_classes, self.nlayers, self.dropout_cla, self.dropout_adj, self.sparse)

        self.optimizer_dat = torch.optim.Adam(self.model_dae.parameters(), lr=self.lr, weight_decay=float(0.0))
        self.optimizer_cla = torch.optim.Adam(self.model_cla.parameters(), lr=self.lr_cla, weight_decay=self.weight_decay)

    def get_loss_reconstruction(self, features, mask):
        if self.dataset == 'ogbn-arxiv' or self.dataset == 'minist' or self.dataset == 'cifar10' or self.dataset == 'fashionmnist':
            masked_features = features * (1 - mask)
            logits, adj = self.model_dae(features, masked_features)

            indices = mask > 0
            loss = F.mse_loss(logits[indices], features[indices], reduction='mean')

            return loss, adj

        logits, adj = self.model_dae(features, features)

        indices = mask > 0
        loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')

        return loss, adj
    
    def get_loss_classification(self, features, adj, mask, labels):

        logits = self.model_cla(features, adj)
        logits = F.log_softmax(logits, 1)

        loss = F.nll_loss(logits[mask], labels[mask], reduction='mean')
        return loss

    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        best_val, best_test = float('-inf'), 0
        hom_ratio_val = 0
        p = 0
        for epoch in range(1, self.num_epochs + 1):
            self.model_dae.train()
            self.model_cla.train()

            self.optimizer_dat.zero_grad()
            self.optimizer_cla.zero_grad()

            mask = get_random_mask(features, self.ratio, self.scale, self.dataset).to(self.device)

            if epoch < self.epochs_pre:
                loss_dae, adj = self.get_loss_reconstruction(features, mask)
                loss_cla = torch.tensor(0).to(self.device)
                loss_hom = torch.tensor(0).to(self.device)
            elif epoch < self.epochs_pre + self.epochs_hom:
                loss_dae, adj = self.get_loss_reconstruction(features, mask)
                loss_cla = self.get_loss_classification(features, adj, train_mask, labels)
                loss_hom = torch.tensor(0).to(self.device)
            else:
                loss_dae, adj = self.get_loss_reconstruction(features, mask)
                loss_cla = self.get_loss_classification(features, adj, train_mask, labels)
                loss_hom = self.model_dae.get_loss_homophily(adj, logits, labels, train_mask, self.nclass, self.num_hop, self.sparse)

            loss = loss_dae * self.alpha + loss_cla + loss_hom * self.beta
            loss.backward()

            self.optimizer_dat.step()
            self.optimizer_cla.step()

            self.model_dae.eval()
            self.model_cla.eval()
            adj = self.model_dae.get_adj(features)
            unnorm_adj = self.model_dae.get_unnorm_adj(features)
            logits = self.model_cla(features, adj)

            train_result = self.metric(logits[train_mask], labels[train_mask])
            val_result = self.metric(logits[val_mask], labels[val_mask])
            test_result = self.metric(logits[test_mask], labels[test_mask])
            hom_ratio = get_homophily(adj, labels, self.sparse).item()

            if epoch >= self.epochs_pre:
                if val_result > best_val:
                    best_val = val_result
                    best_test = test_result
                    hom_ratio_val = hom_ratio
                    p = 0
                    torch.save(unnorm_adj, 'HESGSL_Cora.pt')
                else:
                    p += 1
                    if p >= self.patience:
                        print("Early stopping!")
                        break
            
            if epoch % self.eval_step == 0:
                print(f'Epoch: {epoch: 02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Homophily: {hom_ratio:.2f}',
                      f'Train: {100 * train_result:.2f}%, '
                      f'Valid: {100 * val_result:.2f}%, '
                      f'Test: {100 * test_result:.2f}%')

        print('Best Test Result: ', best_test.item())
        self.best_result = best_test.item()

# GSL.learner.mlp_learner.py
class GSL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayers, k, sparse):
        super(GSL, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hid_dim))
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hid_dim, hid_dim))
        self.layers.append(nn.Linear(hid_dim, out_dim))

        self.k = k
        self.sparse = sparse
        self.in_dim = in_dim
        self.mlp_knn_init()

    def mlp_knn_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.in_dim))

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                h = F.relu(h)
        
        if self.sparse == 1:
            rows, cols, values = knn_fast(h, self.k, 1000)

            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = F.relu(torch.cat((values, values)))

            adj = dgl.graph((rows_, cols_), num_nodes=h.shape[0], device=h.device)
            adj.edata['w'] = values_
        else:
            embeddings = F.normalize(h, dim=1, p=2)
            adj = torch.mm(embeddings, embeddings.t())

            adj = top_k(adj, self.k + 1)
            adj = F.relu(adj)

        return adj


class GCN_DAE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, nlayers, dropout_cla, dropout_adj, mlp_dim, k, sparse):
        super(GCN_DAE, self).__init__()

        self.layers = nn.ModuleList()
        if sparse == 1:
            self.layers.append(GCNConv_dgl(in_dim, hid_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(hid_dim, hid_dim))
            self.layers.append(GCNConv_dgl(hid_dim, out_dim))

            self.dropout_cla = dropout_cla
            self.dropout_adj = dropout_adj
            
        else:
            self.layers = nn.ModuleList()
            self.layers.append(GCNConv(in_dim, hid_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv(hid_dim, hid_dim))
            self.layers.append(GCNConv(hid_dim, out_dim))

            self.dropout_cla = dropout_cla
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.sparse = sparse
        self.graph_generator = GSL(in_dim, math.floor(math.sqrt(in_dim * mlp_dim)), mlp_dim, nlayers, k, sparse)

    def get_adj(self, features):
        if self.sparse == 1:
            return self.graph_generator(features)
        else:
            adj = self.graph_generator(features)
            adj = (adj + adj.T) / 2

            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + 1e-10)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        
    def get_unnorm_adj(self, features):
        if self.sparse == 1:
            return self.graph_generator(features)
        else:
            adj = self.graph_generator(features)
            adj = (adj + adj.T) / 2
            return adj

    def forward(self, features, x):
        adj = self.get_adj(features)
        if self.sparse == 1:
            adj_dropout = adj
            adj_dropout.edata['w'] = F.dropout(adj_dropout.edata['w'], p=self.dropout_adj, training=self.training)
        else:
            adj_dropout = self.dropout_adj(adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, adj_dropout)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_cla, training=self.training)
        x = self.layers[-1](x, adj_dropout)
        
        return x, adj

    def get_loss_homophily(self, g, logits, labels, train_mask, nclasses, num_hop, sparse):

        logits = torch.argmax(logits, dim=-1, keepdim=True)
        logits[train_mask, 0] = labels[train_mask]

        preds = torch.zeros(logits.shape[0], nclasses).to(labels.device)
        preds = preds.scatter(1, logits, 1).detach()

        if sparse == 1:
            g.ndata['l'] = preds
            for _ in range(num_hop):
                g.update_all(fn.u_mul_e('l', 'w', 'm'), fn.sum(msg='m', out='l'))
            q_dist = F.log_softmax(g.ndata['l'], dim=-1)
        else:
            q_dist = preds
            for _ in range(num_hop):
                q_dist = torch.matmul(g, q_dist)
            q_dist = F.log_softmax(q_dist, dim=-1)

        loss_hom = F.kl_div(q_dist, preds)

        return loss_hom