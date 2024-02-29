import torch
import copy
import torch.nn.functional as F
from GSL.encoder import GCN
from GSL.utils import accuracy


class ClsEvaluator:
    def __init__(self, model, config, nfeat, nclass, device):
        self.config = config

        if model == "GCN":
            self.model = GCN(nfeat, self.config.num_hidden_eval, nclass, self.config.num_layers_eval,
                             self.config.dropout_eval, self.config.dropedge_rate_eval, self.config.sparse)
            self.model = self.model.to(device)
        else:
            raise Exception("We don't support the GNN model")

    def compute_loss_acc(self, model, features, adj, mask, labels):
        logits = model(features, adj)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        acc = accuracy(logits[mask], labels[mask])
        return loss, acc


    def __call__(self, features, adj, train_mask, val_mask, test_mask, labels):
        if self.config.sparse:
            f_adj = copy.deepcopy(adj)
            f_adj.edata['w'] = adj.edata['w'].detach()
        else:
            f_adj = adj.detach()
        # self.model.init_para()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_cls, weight_decay=self.config.weight_decay_cls)

        best_val = 0
        best_model = None

        for epoch in range(1, self.config.num_epochs_cls+1):
            self.model.train()
            loss_train, _ = self.compute_loss_acc(self.model, features, f_adj, train_mask, labels)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if epoch % 10 == 0:
                self.model.eval()
                loss_val, acc_val = self.compute_loss_acc(self.model, features, f_adj, val_mask, labels)

                if acc_val > best_val:
                    best_val = acc_val
                    best_model = copy.deepcopy(self.model)

        best_model.eval()
        _, acc_test = self.compute_loss_acc(best_model, features, f_adj, test_mask, labels)

        return {
            'Acc_val': best_val.item(),
            'Acc_test': acc_test.item()
        }
