from GSL.model import BaseModel
from GSL.utils import *
from GSL.learner import *
from GSL.encoder import *
from GSL.metric import *
from GSL.processor import *
from GSL.eval import ClsEvaluator
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F


class SUBLIME(BaseModel):
    '''
    Towards Unsupervised Deep Graph Structure Learning (WWW 2022')
    '''
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device):
        super(SUBLIME, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device)
        self.num_features = num_features
        self.num_classes = num_classes

    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask


        if self.config.mode == 'structure_inference':
            if self.config.sparse:
                anchor_adj_raw = torch_sparse_eye(features.shape[0])
            else:
                anchor_adj_raw = torch.eye(features.shape[0])
        elif self.config.mode == 'structure_refinement':
            # if self.config.sparse:
            #     anchor_adj_raw = adj
            # else:
            #     anchor_adj_raw = torch.from_numpy(adj)
            anchor_adj_raw = adj

        anchor_adj = normalize(anchor_adj_raw, 'sym', self.config['sparse'])

        if self.config.sparse:
            anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
            anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

        activation = ({'relu': F.relu, 'prelu': F.prelu, 'tanh': F.tanh})[self.config.activation]

        #TODO: Graph Learner Initialize
        metric = CosineSimilarity()
        knnsparsify = KNNSparsify(self.config.k)
        processors = [knnsparsify]
        if self.config.learner == 'mlp':
            graph_learner = MLPLearner(metric, processors, 2, features.shape[1], features.shape[1], activation, self.config.sparse, self.config.k)
        elif self.config.learner == 'full':
            graph_learner = FullParam(metric, processors, features.cpu(), self.config.sparse)
        elif self.config.learner == 'gnn':
            graph_learner = GNNLearner(metric, processors, 2, features.shape[1], features.shape[1], activation)
        elif self.config.learner == 'att':
            graph_learner = AttLearner(metric, processors, 2, features.shape[1], activation, self.config.sparse)

        model = GCL(nlayers=self.config.num_layers, in_dim=self.num_features, hidden_dim=self.config.num_hidden,
                         emb_dim=self.config.num_rep_dim, proj_dim=self.config.num_proj_dim,
                         dropout=self.config.dropout, dropout_adj=self.config.dropedge_rate, sparse=self.config.sparse)

        optimizer_cl = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        model = model.to(self.device)
        graph_learner = graph_learner.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)
        if not self.config.sparse:
            anchor_adj = anchor_adj.to(self.device)

        best_test_acc, best_val_acc = 0, 0
        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            graph_learner.train()
            loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj)

            optimizer_cl.zero_grad()
            optimizer_learner.zero_grad()
            loss.backward()
            optimizer_cl.step()
            optimizer_learner.step()

            # Structure Bootstrapping
            if (1 - self.config.tau) and (self.config.c == 0 or epoch % self.config.c == 0):
                if self.config.sparse:
                    learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                    anchor_adj_torch_sparse = anchor_adj_torch_sparse * self.config.tau \
                                                + learned_adj_torch_sparse * (1 - self.config.tau)
                    anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                else:
                    anchor_adj = anchor_adj * self.config.tau + Adj.detach() * (1 - self.config.tau)

            # Evaluate
            if epoch % 20 == 0:
                ClsEval = ClsEvaluator('GCN', self.config, self.num_features, self.num_classes, self.device)
                result = ClsEval(features, anchor_adj, train_mask, \
                                            val_mask, test_mask, labels)
                val_acc, test_acc = result['Acc_val'], result['Acc_test']
                if val_acc > best_val_acc:
                    self.best_test_acc = test_acc
                    best_val_acc = val_acc
                    self.Adj = anchor_adj

                print(f'Epoch: {epoch: 02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Valid: {100 * val_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')



    def loss_gcl(self, model, graph_learner, features, anchor_adj):

        # view 1: anchor graph
        if self.config.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, self.config.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if self.config.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, self.config.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        if not self.config.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', self.config.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if self.config.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.config.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj


class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GCL, self).__init__()

        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse)

    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1