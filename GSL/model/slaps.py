from GSL.model import BaseModel
from GSL.utils import *
from GSL.learner import *
from GSL.encoder import *
from GSL.metric import *
from GSL.processor import *
from GSL.eval import ClsEvaluator
import math
import torch
import torch.nn as nn
from copy import deepcopy

class SLAPS(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, features):
        super(SLAPS, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device)
        self.features = features
        self.Adj = None

        self.model1 = GCN_DAE(config=self.config, nlayers=self.config.nlayers_adj, in_dim=num_features, hidden_dim=self.config.hidden_adj,
                         num_classes=num_features,
                         dropout=self.config.dropout1, dropout_adj=self.config.dropout_adj1,
                         features=features.cpu(), k=self.config.k, knn_metric=self.config.knn_metric, i_=self.config.i,
                         non_linearity=self.config.non_linearity, normalization=self.config.normalization,
                         mlp_h=self.config.mlp_h,
                         mlp_epochs=self.config.mlp_epochs, gen_mode=self.config.gen_mode, sparse=self.config.sparse,
                         mlp_act=self.config.mlp_act).to(device)
        self.model2 = GCN_C(in_channels=num_features, hidden_channels=self.config.hidden, out_channels=num_classes,
                       num_layers=self.config.nlayers, dropout=self.config.dropout2,
                       dropout_adj=self.config.dropout_adj2,
                       sparse=self.config.sparse).to(device)

    def half_val_as_train(self, val_mask, train_mask):
        val_size = np.count_nonzero(val_mask)
        counter = 0
        for i in range(len(val_mask)):
            if val_mask[i] and counter < val_size / 2:
                counter += 1
                val_mask[i] = False
                train_mask[i] = True
        return val_mask, train_mask

    def get_loss_masked_features(self, model, features, mask, ogb, noise, loss_t):
        if ogb:
            if noise == 'mask':
                masked_features = features * (1 - mask)
            elif noise == "normal":
                noise = torch.normal(0.0, 1.0, size=features.shape).to(self.device)
                masked_features = features + (noise * mask)

            logits, Adj = model(features, masked_features)
            indices = mask > 0

            if loss_t == 'bce':
                features_sign = torch.sign(features).to(self.device) * 0.5 + 0.5
                loss = F.binary_cross_entropy_with_logits(logits[indices], features_sign[indices], reduction='mean')
            elif loss_t == 'mse':
                loss = F.mse_loss(logits[indices], features[indices], reduction='mean')
        else:
            masked_features = features * (1 - mask)
            logits, Adj = model(features, masked_features)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(logits[indices], features[indices], reduction='mean')
        return loss, Adj

    def get_loss_learnable_adj(self, model, mask, features, labels, Adj):
        logits = model(features, Adj)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def fit(self, data, split_num=0):
        features, labels = data.features, data.labels
        if data.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = data.train_masks[split_num % 10]
            val_mask = data.val_masks[split_num % 10]
            test_mask = data.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        if self.config.half_val_as_train:
            val_mask, train_mask = self.half_val_as_train(val_mask, train_mask)

        for trial in range(self.config.ntrials):
            optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.config.lr_adj, weight_decay=self.config.w_decay_adj)
            optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.config.lr, weight_decay=self.config.w_decay)

            best_val_result = 0.0

            for epoch in range(1, self.config.epochs_adj + 1):
                self.model1.train()
                self.model2.train()

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                if self.config.dataset.startswith('ogb') or self.config.dataset in ["wine", "digits", "breast_cancer"]:
                    mask = get_random_mask_ogb(features, self.config.ratio).to(self.device)
                    ogb = True
                elif self.config.dataset == "20news10":
                    mask = get_random_mask(features, self.config.ratio, self.config.nr).to(self.device)
                    ogb = True
                else:
                    mask = get_random_mask(features, self.config.ratio, self.config.nr).to(self.device)
                    ogb = False

                if epoch < self.config.epochs_adj // self.config.epoch_d:
                    self.model2.eval()
                    loss1, Adj = self.get_loss_masked_features(self.model1, features, mask, ogb, self.config.noise, self.config.loss)
                    loss2 = torch.tensor(0).to(self.device)
                else:
                    loss1, Adj = self.get_loss_masked_features(self.model1, features, mask, ogb, self.config.noise, self.config.loss)
                    loss2, train_result = self.get_loss_learnable_adj(self.model2, train_mask, features, labels, Adj)

                loss = loss1 * self.config.lambda_ + loss2
                loss.backward()
                optimizer1.step()
                optimizer2.step()

                if epoch >= self.config.epochs_adj // self.config.epoch_d and epoch % 1 == 0:
                    val_result = self.test(val_mask, features, labels, Adj)
                    test_result = self.test(test_mask, features, labels, Adj)
                    if val_result > best_val_result:
                        best_val_result = val_result
                        self.best_result = test_result
                        self.Adj = Adj
                    print(f'Epoch: {epoch: 02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_result.item():.2f}%, '
                          f'Valid: {100 * val_result:.2f}%, '
                          f'Test: {100 * test_result:.2f}%')

    def test(self, test_mask, features, labels, adj):
        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()
            test_loss_, test_accu_ = self.get_loss_learnable_adj(self.model2, test_mask, features, labels, adj)
            return test_accu_.item()

# TODO: move GCN_DAE and GCN_C to GSL/encoder/
class GCN_DAE(nn.Module):
    def __init__(self, config, nlayers, in_dim, hidden_dim, num_classes, dropout, dropout_adj, features, k, knn_metric, i_,
                 non_linearity, normalization, mlp_h, mlp_epochs, gen_mode, sparse, mlp_act):
        super(GCN_DAE, self).__init__()

        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.layers.append(GCNConv_dgl(hidden_dim, num_classes))

        else:
            self.layers.append(GCNConv(in_dim, hidden_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layers.append(GCNConv(hidden_dim, num_classes))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.k = k
        self.knn_metric = knn_metric
        self.i = i_
        self.non_linearity = non_linearity
        self.normalization = normalization
        self.nnodes = features.shape[0]
        self.mlp_h = mlp_h
        self.mlp_epochs = mlp_epochs
        self.sparse = sparse

        if gen_mode == 0:
            metric = CosineSimilarity()
            processors = [KNNSparsify(k), Discretize(), LinearTransform(config.i)]
            non_linearize = NonLinearize(config.non_linearity, alpha=config.i)
            self.graph_gen = FullParam(metric, processors, features, sparse, non_linearize)
        elif gen_mode == 1:
            metric = CosineSimilarity()
            processors = [KNNSparsify(k)]
            activation = ({'relu': F.relu, 'prelu': F.prelu, 'tanh': F.tanh})[mlp_act]
            self.graph_gen = MLPLearner(metric, processors, 2, features.shape[1],
                                        math.floor(math.sqrt(features.shape[1] * self.mlp_h)), activation, sparse, k=k)
        # TODO: implement MLP-D in PyGSL style
        # elif gen_mode == 2:
        #     self.graph_gen = MLP_Diag(2, features.shape[1], k, knn_metric, self.non_linearity, self.i, sparse,
        #                               mlp_act).to(self.device)

    def get_adj(self, h):
        Adj_ = self.graph_gen(h)
        if not self.sparse:
            Adj_ = symmetrize(Adj_)
            Adj_ = normalize(Adj_, self.normalization, self.sparse)
        return Adj_

    def forward(self, features, x):  # x corresponds to masked_fearures
        Adj_ = self.get_adj(features)
        if self.sparse:
            Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x, Adj_



class GCN_C(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GCN_C, self).__init__()

        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.layers.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

    def forward(self, x, adj_t):
        if self.sparse:
            Adj = adj_t
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(adj_t)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x
