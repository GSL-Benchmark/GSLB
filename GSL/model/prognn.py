import time
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.optimizer import required

from GSL.learner import FullParam
from GSL.model import BaseModel
from GSL.processor import NonLinearize, Normalize
from GSL.utils import accuracy


class ProxOperators:
    """Proximal Operators."""

    def __init__(self):
        self.nuclear_norm = None

    def prox_l1(self, data, alpha):
        """Proximal operator for l1 norm."""
        data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data) - alpha, min=0))
        return data

    def prox_nuclear(self, data, alpha):
        """Proximal operator for nuclear norm (trace norm)."""
        device = data.device
        U, S, V = np.linalg.svd(data.cpu())
        U, S, V = (
            torch.FloatTensor(U).to(device),
            torch.FloatTensor(S).to(device),
            torch.FloatTensor(V).to(device),
        )
        self.nuclear_norm = S.sum()
        # print("nuclear norm: %.4f" % self.nuclear_norm)

        diag_S = torch.diag(torch.clamp(S - alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_truncated_2(self, data, alpha, k=50):
        device = data.device
        import tensorly as tl

        tl.set_backend("pytorch")
        U, S, V = tl.truncated_svd(data.cpu(), n_eigenvecs=k)
        U, S, V = (
            torch.FloatTensor(U).to(device),
            torch.FloatTensor(S).to(device),
            torch.FloatTensor(V).to(device),
        )
        self.nuclear_norm = S.sum()
        # print("nuclear norm: %.4f" % self.nuclear_norm)

        S = torch.clamp(S - alpha, min=0)

        # diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        # U = torch.spmm(U, diag_S)
        # V = torch.matmul(U, V)

        # make diag_S sparse matrix
        indices = torch.tensor((range(0, len(S)), range(0, len(S)))).to(device)
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size((len(S), len(S))))
        V = torch.spmm(diag_S, V)
        V = torch.matmul(U, V)
        return V

    def prox_nuclear_truncated(self, data, alpha, k=50):
        device = data.device
        indices = torch.nonzero(data).t()
        values = data[indices[0], indices[1]]  # modify this based on dimensionality
        data_sparse = sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()))
        U, S, V = sp.linalg.svds(data_sparse, k=k)
        U, S, V = (
            torch.FloatTensor(U).to(device),
            torch.FloatTensor(S).to(device),
            torch.FloatTensor(V).to(device),
        )
        self.nuclear_norm = S.sum()
        diag_S = torch.diag(torch.clamp(S - alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_cuda(self, data, alpha):

        device = data.device
        U, S, V = torch.svd(data)
        # self.nuclear_norm = S.sum()
        # print(f"rank = {len(S.nonzero())}")
        self.nuclear_norm = S.sum()
        S = torch.clamp(S - alpha, min=0)
        indices = torch.tensor([range(0, U.shape[0]), range(0, U.shape[0])]).to(device)
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size(U.shape))
        # diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        # print(f"rank_after = {len(diag_S.nonzero())}")
        V = torch.spmm(diag_S, V.t_())
        V = torch.matmul(U, V)
        return V


prox_operators = ProxOperators()


class ProGNN(BaseModel):
    """
    Graph Structure Learning for Robust Graph Neural Networks (KDD 2020')
    """

    def __init__(self, model, config, device):
        super().__init__(device=device)
        self.device = device
        self.config = config
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.model = model.to(device)

    def fit(self, features, adj, labels, idx_train, idx_val, **kwargs):
        config = self.config

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.normalizer = Normalize()
        non_linearize = NonLinearize('none')
        estimator = FullParam(adj=adj, non_linearize=non_linearize)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(
            estimator.parameters(), momentum=0.9, lr=config["lr"]
        )

        self.optimizer_l1 = PGD(
            estimator.parameters(),
            proxs=[prox_operators.prox_l1],
            lr=config["lr_adj"],
            alphas=[config["alpha"]],
        )

        self.optimizer_nuclear = PGD(
            estimator.parameters(),
            proxs=[prox_operators.prox_nuclear_cuda],
            lr=config["lr_adj"],
            alphas=[config["beta"]],
        )

        # Train model
        t_total = time.time()
        for epoch in range(config["epochs"]):
            for i in range(int(config["outer_steps"])):
                self.train_adj(epoch, features, adj, labels, idx_train, idx_val)

            for i in range(int(config["inner_steps"])):
                self.train_gcn(
                    epoch,
                    features,
                    estimator.adj,
                    labels,
                    idx_train,
                    idx_val,
                )

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(config)

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        config = self.config
        estimator = self.estimator
        adj = self.normalizer(estimator.adj + torch.eye(adj.shape[0]).to(self.device))
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if config["debug"]:
                print(
                    "\t=== saving current graph/gcn, best_val_acc: %s"
                    % self.best_val_acc.item()
                )

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if config["debug"]:
                print(
                    "\t=== saving current graph/gcn, best_val_loss: %s"
                    % self.best_val_loss.item()
                )

        if config["debug"]:
            if epoch % 1 == 0:
                print(
                    "Epoch: {:04d}".format(epoch + 1),
                    "loss_train: {:.4f}".format(loss_train.item()),
                    "acc_train: {:.4f}".format(acc_train.item()),
                    "loss_val: {:.4f}".format(loss_val.item()),
                    "acc_val: {:.4f}".format(acc_val.item()),
                    "time: {:.4f}s".format(time.time() - t),
                )

    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val):
        estimator = self.estimator
        config = self.config
        if config["debug"]:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.adj, 1)
        loss_fro = torch.norm(estimator.adj - adj, p="fro")
        normalized_adj = self.normalizer(estimator.adj + torch.eye(adj.shape[0]).to(self.device))

        if config["lambda_"]:
            loss_smooth_feat = self.feature_smoothing(estimator.adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        output = self.model(features, normalized_adj)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_symmetric = torch.norm(
            estimator.adj - estimator.adj.t(), p="fro"
        )

        loss_diffiential = (
            loss_fro
            + config["gamma"] * loss_gcn
            + config["lambda_"] * loss_smooth_feat
            + config["phi"] * loss_symmetric
        )

        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear = 0 * loss_fro
        if config["beta"] != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = (
            loss_fro
            + config["gamma"] * loss_gcn
            + config["alpha"] * loss_l1
            + config["beta"] * loss_nuclear
            + config["phi"] * loss_symmetric
        )

        estimator.adj.data.copy_(
            torch.clamp(estimator.adj.data, min=0, max=1)
        )

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.']
        self.model.eval()
        normalized_adj = self.normalizer(estimator.adj + torch.eye(adj.shape[0]).to(self.device))
        output = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "acc_train: {:.4f}".format(acc_train.item()),
            "loss_val: {:.4f}".format(loss_val.item()),
            "acc_val: {:.4f}".format(acc_val.item()),
            "time: {:.4f}s".format(time.time() - t),
        )

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if config["debug"]:
                print(
                    "\t=== saving current graph/gcn, best_val_acc: %s"
                    % self.best_val_acc.item()
                )

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if config["debug"]:
                print(
                    "\t=== saving current graph/gcn, best_val_loss: %s"
                    % self.best_val_loss.item()
                )

        if config["debug"]:
            if epoch % 1 == 0:
                print(
                    "Epoch: {:04d}".format(epoch + 1),
                    "loss_fro: {:.4f}".format(loss_fro.item()),
                    "loss_gcn: {:.4f}".format(loss_gcn.item()),
                    "loss_feat: {:.4f}".format(loss_smooth_feat.item()),
                    "loss_symmetric: {:.4f}".format(loss_symmetric.item()),
                    "delta_l1_norm: {:.4f}".format(
                        torch.norm(estimator.adj - adj, 1).item()
                    ),
                    "loss_l1: {:.4f}".format(loss_l1.item()),
                    "loss_total: {:.4f}".format(total_loss.item()),
                    "loss_nuclear: {:.4f}".format(loss_nuclear.item()),
                )

    def test(self, features, labels, idx_test, hetero=False):
        """Evaluate the performance of ProGNN on test set"""
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.normalizer(self.estimator.adj + torch.eye(adj.shape[0]).to(self.device))
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        if hetero:
            pred = torch.argmax(output, dim=1)
            return torch_f1_score(pred[idx_test], labels[idx_test])
        else:
            acc_test = accuracy(output[idx_test], labels[idx_test])
            print(
                "\tTest set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()),
            )
            return acc_test.item()

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj) / 2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat

class PGD(Optimizer):
    """Proximal gradient descent.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining parameter groups
    proxs : iterable
        iterable of proximal operators
    alpha : iterable
        iterable of coefficients for proximal gradient descent
    lr : float
        learning rate
    momentum : float
        momentum factor (default: 0)
    weight_decay : float
        weight decay (L2 penalty) (default: 0)
    dampening : float
        dampening for momentum (default: 0)

    """

    def __init__(
        self,
        params,
        proxs,
        alphas,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
    ):
        defaults = dict(lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)

        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault("proxs", proxs)
            group.setdefault("alphas", alphas)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            # group.setdefault("proxs", proxs)
            # group.setdefault("alphas", alphas)

    def step(self, delta=0, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            proxs = group["proxs"]
            alphas = group["alphas"]

            # apply the proximal operator to each parameter in a group
            for param in group["params"]:
                for prox_operator, alpha in zip(proxs, alphas):
                    # param.data.add_(lr, -param.grad.data)
                    # param.data.add_(delta)
                    param.data = prox_operator(param.data, alpha=alpha * lr)
