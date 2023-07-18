from GSL.model import BaseModel
from GSL.encoder import GCN
# from GSL.utils import row_normalize

import torch
import torch.nn.functional as F


class GCN_Trainer(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device):
        super(GCN_Trainer, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device)
        self.gcn = GCN(in_channels=num_features,
                       hidden_channels=self.config.hidden,
                       out_channels=num_classes,
                       num_layers=2,
                       dropout=self.config.dropout,
                       dropout_adj=0,
                       sparse=self.config.sparse)
        
    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        features = row_normalize(features)
        adj = row_normalize(adj)
        
        optimizer = torch.optim.Adam(self.gcn.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        best_val, best_result = float("-inf"), 0
        for epoch in range(self.config.epochs):
            self.gcn.train()
            optimizer.zero_grad()

            output = self.gcn(features, adj)
            loss = F.cross_entropy(output[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.gcn.eval()
                output = self.gcn(features, adj)
                train_result = self.metric(output[train_mask], labels[train_mask])
                val_result = self.metric(output[val_mask], labels[val_mask])
                test_result = self.metric(output[test_mask], labels[test_mask])
            
            print(f'Epoch: {epoch: 02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_result:.2f}%, '
                  f'Valid: {100 * val_result:.2f}%, '
                  f'Test: {100 * test_result:.2f}%')
            
            if val_result > best_val:
                best_val = val_result
                best_result = test_result

        print('Best Test Result: ', best_result.item())
        self.best_result = best_result.item()


class MLP_Trainer(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device):
        super(MLP_Trainer, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device)
        self.linear1 = torch.nn.Linear(num_features, self.config.hidden)
        self.linear2 = torch.nn.Linear(self.config.hidden, num_classes)

    def feedforward(self, x, adj=None):
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        return x
    
    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        best_val, best_result = float("-inf"), 0
        for epoch in range(self.config.epochs):
            self.train()
            optimizer.zero_grad()

            output = self.feedforward(features)
            loss = F.cross_entropy(output[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.feedforward(features)
                train_result = self.metric(output[train_mask], labels[train_mask])
                val_result = self.metric(output[val_mask], labels[val_mask])
                test_result = self.metric(output[test_mask], labels[test_mask])
            
            print(f'Epoch: {epoch: 02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_result:.2f}%, '
                  f'Valid: {100 * val_result:.2f}%, '
                  f'Test: {100 * test_result:.2f}%')
            
            if val_result > best_val:
                best_val = val_result
                best_result = test_result

        print('Best Test Result: ', best_result.item())
        self.best_result = best_result.item()