from GSL.model import BaseModel
from GSL.learner import SpLearner
from GSL.encoder import GCN
from GSL.utils import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm


class NeuralSparse(BaseModel):
    """
    Robust Graph Representation Learning via Neural Sparsification (ICML 2020)
    """
    def __init__(self, config, device, num_features, num_classes):
        super(NeuralSparse, self).__init__(device=device)
        self.config = config
        self.learner = SpLearner(nlayers=2, in_dim=2*num_features+1, hidden=config.hidden, activation=nn.ReLU(), k=config.k)
        self.gnn = GCN(in_channels=num_features,
                       hidden_channels=config.hidden_channel,
                       out_channels=num_classes,
                       num_layers=config.num_layers,
                       dropout=config.dropout,
                       dropout_adj=0,
                       sparse=False)
        self.optimizer = torch.optim.Adam(list(self.learner.parameters())+list(self.gnn.parameters()), lr=config.lr, weight_decay=config.weight_decay)

    def test(self, features, adj, labels, mask):
        self.learner.eval()
        self.gnn.eval()
        with torch.no_grad():
            support = self.learner(features, adj, 1.0, False)
            output = self.gnn(features, support)
        acc = accuracy(output[mask], labels[mask]).item()
        return acc


    def fit(self, features, adj, labels, train_mask, val_mask, test_mask):
        features = features.to(self.device)
        labels = labels.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        adj = torch.tensor(adj).to(self.device)

        indices = torch.nonzero(adj)
        values = adj[indices[:, 0], indices[:, 1]]
        shape = adj.shape

        adj = torch.sparse_coo_tensor(indices.t(), values, torch.Size(shape)).to(torch.float32)

        best_val_acc, best_test_acc = 0, 0
        with tqdm(total=self.config.epochs, desc='(NeuralSparse)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.config.epochs):
                self.learner.train()
                self.gnn.train()
                if epoch % self.config.temp_N == 0:
                    decay_temp = np.exp(-1*self.config.temp_r*epoch)
                    temp = max(0.05, decay_temp)
                support = self.learner(features, adj, temp, True)
                output = self.gnn(features, support)
                loss_train = F.cross_entropy(output[train_mask], labels[train_mask])

                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

                val_acc = self.test(features, adj, labels, val_mask)
                test_acc = self.test(features, adj, labels, test_mask)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    patience = 0
                else:
                    patience += 1

                if patience > self.config.patience:
                    print("Early stopping...")
                    break

                pbar.set_postfix({'Epoch': epoch+1, 'Loss': loss_train.item(), 'Acc_val': val_acc, 'Acc_test': test_acc})
                pbar.update(1)
        print("Best Test Accuracy: ", best_test_acc)