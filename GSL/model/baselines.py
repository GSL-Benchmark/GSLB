from GSL.model import BaseModel
from GSL.encoder import GCN, GPRGNN, DAGNN
from GSL.utils import *

import torch
import torch.nn.functional as F


class GCN_Trainer(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, params):
        super(GCN_Trainer, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device, params)
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

        features = row_normalize_features(features)
        adj = row_normalize_features(adj)
        
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
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, params):
        super(MLP_Trainer, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device, params)
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


class GPRGNN_Trainer(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, params):
        super(GPRGNN_Trainer, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device, params)
        self.model = GPRGNN(num_feat=num_features,
                            num_class=num_classes,
                            hidden=self.config.hidden,
                            ppnp=self.config.ppnp,
                            K=self.config.K,
                            alpha=self.config.alpha,
                            Init=self.config.Init,
                            Gamma=self.config.Gamma,
                            dprate=self.config.dprate,
                            dropout=self.config.dropout)
    
    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        g = adjacency_matrix_to_dgl(adj)
        g = g.add_self_loop()
        g = g.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        best_val, best_result = float("-inf"), 0
        for epoch in range(self.config.epochs):
            self.train()
            optimizer.zero_grad()

            output = self.model(g, features)
            loss = F.cross_entropy(output[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.model(g, features)
                train_result = self.metric(output[train_mask], labels[train_mask])
                val_result = self.metric(output[val_mask], labels[val_mask])
                test_result = self.metric(output[test_mask], labels[test_mask])
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch: 02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_result:.2f}%, '
                    f'Valid: {100 * val_result:.2f}%, '
                    f'Test: {100 * test_result:.2f}%')
            
            if val_result > best_val:
                best_val = val_result
                best_result = test_result

        self.best_result = best_result.item()


class DAGNN_Trainer(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, params):
        super(DAGNN_Trainer, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device, params)
        self.model = DAGNN(k=self.config.K,
                           in_dim=num_features,
                           hid_dim=self.config.hidden,
                           out_dim=num_classes,
                           dropout=self.config.dropout)
        
    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        g = adjacency_matrix_to_dgl(adj)
        g = g.add_self_loop()
        g = g.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        best_val, best_result = float("-inf"), 0
        for epoch in range(self.config.epochs):
            self.train()
            optimizer.zero_grad()

            output = self.model(g, features)
            loss = F.cross_entropy(output[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                output = self.model(g, features)
                train_result = self.metric(output[train_mask], labels[train_mask])
                val_result = self.metric(output[val_mask], labels[val_mask])
                test_result = self.metric(output[test_mask], labels[test_mask])
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch: 02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_result:.2f}%, '
                    f'Valid: {100 * val_result:.2f}%, '
                    f'Test: {100 * test_result:.2f}%')
            
            if val_result > best_val:
                best_val = val_result
                best_result = test_result

        self.best_result = best_result.item()


class Self_Training(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, params):
        super(Self_Training, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device, params)
        self.model_path = './self_training.pth'
        self.best_val = float("inf")

    def build_model(self):
        model = GCN(in_channels=self.num_feat,
                    hidden_channels=self.config.hidden,
                    out_channels=self.num_class,
                    num_layers=2,
                    dropout=self.config.dropout,
                    dropout_adj=0,
                    sparse=self.config.sparse)
        return model
    
    def generate_trainmask(self, train_mask, labels, labelrate):
        train_index = torch.where(train_mask)[0]
        train_mask = train_mask.clone()
        train_mask[:] = False
        label = labels[train_index]
        for i in range(self.num_class):
            class_index = torch.where(label == i)[0].tolist()
            class_index = random.sample(class_index, labelrate)
            train_mask[train_index[class_index]] = True
        return train_mask
        
    def train(self, train_mask_ag, val_mask, test_mask, features, adj, pseudo_labels, labels, stage):
        model = self.build_model()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()

            output = model(features, adj)
            loss = F.cross_entropy(output[train_mask_ag], pseudo_labels[train_mask_ag])
            train_result = self.metric(output[train_mask_ag], pseudo_labels[train_mask_ag])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                output = model(features, adj)
                val_result = self.metric(output[val_mask], labels[val_mask])
                test_result = self.metric(output[test_mask], labels[test_mask])
                val_loss = F.cross_entropy(output[val_mask], labels[val_mask])

            if val_loss < self.best_val:
                torch.save(model.state_dict(), self.model_path)
                self.best_result = test_result.item()
                self.best_val = val_loss
                bad_counter = 0
            # else:
            #     bad_counter += 1

            # if bad_counter == self.config.patience:
            #     break
            
            if epoch % 100 == 0:
                print(f'Stage: {stage: 02d}, '
                    f'Epoch: {epoch: 02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_result:.2f}%, '
                    f'Valid: {100 * val_result:.2f}%, '
                    f'Test: {100 * test_result:.2f}%')
            
    def get_confidence(self, output, with_softmax=False):
        if not with_softmax:
            output = torch.softmax(output, dim=1)

        confidence, pred_label = torch.max(output, dim=1)

        return confidence, pred_label
    
    def regenerate_pseudo_label(self, output, labels, train_mask, unlabeled_mask, threshold, sign=False):
        unlabeled_index = torch.where(unlabeled_mask == True)[0]
        train_idx = torch.where(train_mask == True)[0]
        confidence, pred_label = self.get_confidence(output, sign)
        index = torch.where(confidence > threshold)[0]
        pseudo_index = []
        pseudo_labels, train_mask_ag = labels.clone(), train_mask.clone()
        for i in index:
            if i not in train_idx:
                pseudo_labels[i] = pred_label[i]
                if i in unlabeled_index:
                    train_mask_ag[i] = True
                    pseudo_index.append(i)
        pseudo_index = torch.tensor(pseudo_index)
        return train_mask_ag, pseudo_labels

                
    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        train_mask = self.generate_trainmask(train_mask, labels, 3)

        features = row_normalize_features(features)
        adj = row_normalize_features(adj)

        train_mask_ag = train_mask.clone()
        pseudo_labels = labels.clone()

        for stage in range(self.config.stages):
            self.train(train_mask_ag, val_mask, test_mask, features, adj, pseudo_labels, labels, stage)
            unlabeled_mask = ~(train_mask | val_mask | test_mask)
            state_dict = torch.load(self.model_path)
            best_model = self.build_model()
            best_model.load_state_dict(state_dict)
            best_model.to(self.device)
            best_model.eval()
            best_output = best_model(features, adj)
            train_mask_ag, pseudo_labels = self.regenerate_pseudo_label(best_output, labels, train_mask, unlabeled_mask,
                                                                                    self.config.threshold)
            
        os.system(f'rm -rf {self.model_path}')
