from GSL.model import BaseModel
from GSL.encoder import GCNConv, GCNConv_dgl
from GSL.utils import accuracy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

eps = 1e-7
SVD_PI = True


class PTDNet(BaseModel):
    """
    Learning to Drop: Robust Graph Neural Network via Topological Denoising (WSDM 2021)
    """
    def __init__(self, config, device, in_dim, out_dim):
        super(PTDNet, self).__init__(device=device)
        self.config = config
        self.sparse = config.sparse
        self.in_dim = in_dim
        self.out_dim = out_dim
        hiddens = config.hiddens

        self.layers = nn.ModuleList()

        try:
            hiddens = [int(s) for s in str(hiddens).split('-')]
        except:
            hiddens = [config.hidden1]

        nhiddens = len(hiddens)

        if self.sparse:
            self.layers.append(GCNConv_dgl(self.in_dim, hiddens[0]))
            for _ in range(1, nhiddens):
                self.layers.append(GCNConv_dgl(hiddens[-1], hiddens[_]))
            self.layers.append(GCNConv_dgl(hiddens[-1], self.out_dim))
        else:
            self.layers.append(GCNConv(self.in_dim, hiddens[0]))
            for i in range(1, nhiddens):
                self.layers.append(GCNConv(hiddens[-1], hiddens[_]))
            self.layers.append(GCNConv(hiddens[-1], self.out_dim))

        hidden_1 = config.denoise_hidden_1
        hidden_2 = config.denoise_hidden_2
        self.edge_weights = []
        self.nblayers = nn.ModuleList()
        self.selflayers = nn.ModuleList()

        self.attentions = nn.ModuleList()
        self.attentions.append(nn.ModuleList())
        for _ in hiddens:
            self.attentions.append(nn.ModuleList())

        self.nblayers.append(nn.Linear(self.in_dim, hidden_1))
        self.selflayers.append(nn.Linear(self.in_dim, hidden_1))

        if hidden_2 > 0:
            self.attentions[0].append(nn.Linear(hidden_1*2, hidden_2))

        self.attentions[0].append(nn.Linear(hidden_1*2, 1))

        for i in range(1, len(self.attentions)):
            self.nblayers.append(nn.Linear(hiddens[i-1], hidden_1))
            self.selflayers.append(nn.Linear(hiddens[i-1], hidden_1))

            if hidden_2>0:
                self.attentions[i].append(nn.Linear(hidden_1*2, hidden_2))

            self.attentions[i].append(nn.Linear(hidden_1*2, 1))

    def get_attention(self, input1, input2, l=0, training=False):

        nb_layer = self.nblayers[l]
        selflayer = self.selflayers[l]
        net = self.attentions[l]

        input1 = F.relu(nb_layer(input1))
        if training:
            input1 = F.dropout(input1, p=self.config.dropout)
        input2 = F.relu(selflayer(input2))
        if training:
            input2 = F.dropout(input2, p=self.config.dropout)

        input10 = torch.cat([input1, input2], dim=1)
        for _layer in net:
            input10 = _layer(input10)
            if training:
                input10 = F.dropout(input10, p=self.config.dropout)
        weight10 = input10
        return weight10

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        gamma = self.config.gamma
        zeta = self.config.zeta

        if training:
            debug_var = eps
            bias = 0.0
            random_noise = bias+torch.empty_like(log_alpha).uniform_(debug_var, 1.0 - debug_var).to(torch.float32)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = torch.clamp(stretched_values, min=0.0, max=1.0)
        return cliped

    def l0_norm(self, log_alpha, beta):
        gamma = self.config.gamma
        zeta = self.config.zeta
        temp = torch.tensor(-gamma / zeta).to(self.device)
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(temp).to(torch.float32))
        return torch.mean(reg_per_weight)

    def lossl0(self, temperature):
        l0_loss = torch.zeros([], dtype=torch.float32).to(self.device)
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)
        self.edge_weights = []
        return l0_loss

    def nuclear(self):
        nuclear_loss = torch.zeros([], dtype=torch.float32)
        values = []
        if self.config.lambda3 == 0:
            return 0
        for mask in self.maskes:
            mask = torch.squeeze(mask)
            support = torch.sparse_coo_tensor(indices=self.indices.t(),
                                              values=mask,
                                              size=self.adj_mat.shape)
            support_dense = support.to_dense()
            support_trans = support_dense.t()

            AA = torch.matmul(support_trans, support_dense)
            if SVD_PI:
                row_ind = self.indices[:, 0].cpu().numpy()
                col_ind = self.indices[:, 1].cpu().numpy()
                support_csc = csc_matrix((mask.detach().cpu().numpy(), (row_ind, col_ind)))
                k = self.config.k_svd
                u, s, vh = svds(support_csc, k=k)

                u = torch.tensor(u.copy())
                s = torch.tensor(s.copy())
                vh = torch.tensor(vh.copy())

                for i in range(k):
                    vi = torch.unsqueeze(vh[i], -1).to(self.device)
                    for ite in range(1):
                        vi = torch.matmul(AA, vi)
                        vi_norm = torch.linalg.norm(vi)
                        vi = vi / vi_norm

                    vmv = torch.matmul(vi.t(), torch.matmul(AA, vi))
                    vv = torch.matmul(vi.t(), vi)

                    t_vi = torch.sqrt(torch.abs(vmv / vv))
                    values.append(t_vi)

                    if k > 1:
                        AA_minus = torch.matmul(AA, torch.matmul(vi, torch.transpose(vi)))
                        AA = AA - AA_minus
            else:
                trace = torch.linalg.trace(AA)
                values.append(torch.sum(trace))

            nuclear_loss = torch.sum(torch.stack(values, dim=0), dim=0)

        return nuclear_loss

    def feedforward(self, inputs, training=None):
        if training:
            temperature = inputs
        else:
            temperature = 1.0

        self.edge_maskes = []
        self.maskes = []
        layer_index = 0
        x = self.features

        for i in range(len(self.layers)):
            xs = []
            layer = self.layers[i]
            for l in range(self.config.L):
                f1_features = torch.index_select(x, 0, self.row)
                f2_features = torch.index_select(x, 0, self.col)
                weight = self.get_attention(f1_features, f2_features, l=layer_index, training=True)
                mask = self.hard_concrete_sample(weight, temperature, True)
                self.edge_weights.append(weight)
                self.maskes.append(mask)
                mask = torch.squeeze(mask)
                adj = torch.sparse_coo_tensor(indices=self.indices.t(),
                                                values=mask,
                                                size=self.adj_mat.shape)
                # norm
                adj = torch.eye(self.node_num, dtype=torch.float32).to_sparse().to(self.device) + adj

                row = adj.coalesce().indices().t()[:, 0]
                col = adj.coalesce().indices().t()[:, 1]

                rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
                d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
                d_inv_sqrt = torch.clamp(d_inv_sqrt, 0, 10.0)
                row_inv_sqrt = torch.index_select(d_inv_sqrt, 0, row)
                col_inv_sqrt = torch.index_select(d_inv_sqrt, 0, col)
                values = torch.mul(adj.coalesce().values(), row_inv_sqrt)
                values = torch.mul(values, col_inv_sqrt)

                support = torch.sparse_coo_tensor(indices=adj.coalesce().indices(),
                                                    values = values,
                                                    size=adj.shape).to_dense().to(torch.float32)
                nextx = layer(x, support)
                if i != len(self.layers)-1:
                    nextx = F.relu(nextx)
                xs.append(nextx)
            x = torch.mean(torch.stack(xs, dim=0), dim=0)
            layer_index += 1
        return x

    def compute_loss(self, preds, temperature, labels, train_mask):
        all_preds = torch.cat(preds, dim=0)
        mean_preds = torch.mean(torch.stack(preds, dim=0), dim=0)
        mean_preds = torch.squeeze(mean_preds, dim=0)
        diff = mean_preds - all_preds

        consistency_loss = F.mse_loss(diff, torch.zeros_like(diff))

        cross_loss = self.criterion(mean_preds[train_mask], labels[train_mask])
        lossl0 = self.lossl0(temperature)
        nuclear = self.nuclear()
        loss = cross_loss + self.config.lambda1*lossl0 + self.config.lambda3*nuclear + self.config.coff_consis*consistency_loss
        return loss

    def test(self, labels, mask):
        with torch.no_grad():
            output = self.feedforward(None, False)
        acc = accuracy(output[mask], labels[mask]).item()
        return acc


    def fit(self, features, adj, labels, train_mask, val_mask, test_mask):
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.config.weight_decay, lr=self.config.lr)
        self.criterion = nn.CrossEntropyLoss()
        labels = labels.to(self.device)

        self.adj = adj.to(self.device)
        self.features = features.to(self.device)
        self.indices = adj.coalesce().indices().t().to(self.device)
        self.node_num = features.shape[0]
        self.values = adj.coalesce().values()
        self.shape = adj.shape
        self.row = self.indices[:, 0]
        self.col = self.indices[:, 1]
        self.adj_mat = adj

        best_val_result, best_test_result = float('-inf'), 0
        patience = 0
        for epoch in range(self.config.epochs):
            temperature = max(0.05, self.config.init_temperature * pow(self.config.temperature_decay, epoch))
            preds = []

            for _ in range(self.config.outL):
                output = self.feedforward(temperature, True)
                preds.append(torch.unsqueeze(output, 0))

            loss = self.compute_loss(preds, temperature, labels, train_mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_result = self.test(labels, train_mask)
            val_result = self.test(labels, val_mask)
            test_result = self.test(labels, test_mask)
            if val_result > best_val_result:
                best_val_result = val_result
                best_test_result = test_result
                patience = 0
            else:
                patience += 1

            print(f'Epoch: {epoch: 02d}, '
                f'Loss: {loss.item():.4f}, '
                f'Train: {100 * train_result:.2f}%, '
                f'Valid: {100 * val_result:.2f}%, '
                f'Test: {100 * test_result:.2f}%')

            if patience > self.config.patience:
                print("Early stopping...")
                break

        print("Best Test Accuracy: ", best_test_result)