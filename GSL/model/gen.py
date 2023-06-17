import time
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity as cos

from GSL.encoder import GCN
from GSL.metric import CosineSimilarity
from GSL.model import BaseModel
from GSL.processor import KNNSparsify
from GSL.utils import accuracy, get_homophily, prob_to_adj, torch_f1_score


class GEN(BaseModel):
    def __init__(self, device, config):
        super().__init__(device)
        self.config = config
        self.device = device
        # self.knn = KNNGraphFromFeature(
        #     k=self.config["k"], metric="minkowski", i=self.config["i"]
        # )
        self.base_model = GCN(
            config["num_feat"],
            config["hidden"],
            config["num_class"],
            num_layers=2,
            dropout=config["dropout"],
            dropout_adj=0.0,
            sparse=False,
            activation_last="log_softmax",
        )
        self.best_acc_val = 0.0
        self.iter = 0
        self.k = config["k"]

    def knn(self, feature):
        adj = np.zeros((self.num_node, self.num_node), dtype=np.int64)
        dist = cos(feature.detach().cpu().numpy())
        col = np.argpartition(dist, -(self.config.k + 1), axis=1)[:,-(self.config.k + 1):].flatten()
        adj[np.arange(self.num_node).repeat(self.config.k + 1), col] = 1
        return adj
        # metric = CosineSimilarity()
        # processor = KNNSparsify(self.k)
        # dist = metric(feature)
        # adj_float = processor(dist).detach().cpu().numpy()
        # return (adj_float > 0).astype(np.int64)

    def fit(self, features, adj, labels, mask_train, mask_val):
        self.num_class = self.config["num_class"]
        self.num_node = self.config["num_node"]

        homophily = get_homophily(label=labels, adj=adj)
        estimator = EstimateAdj(
            features,
            labels,
            adj,
            torch.where(mask_train)[0],
            self.num_class,
            self.num_node,
            homophily,
        )

        t_total = time.time()
        for iter in range(self.config["iter"]):
            self.train_base_model(features, adj, labels, mask_train, mask_val, iter)

            estimator.reset_obs()
            estimator.update_obs(self.knn(features))
            estimator.update_obs(self.knn(self.hidden_output))
            estimator.update_obs(self.knn(self.output))
            self.iter += 1
            alpha, beta, O, Q, iterations = estimator.EM(
                self.output.max(1)[1].detach().cpu().numpy(), self.config["tolerance"]
            )
            adj = prob_to_adj(Q, self.config["threshold"]).to(self.device)

        print(
            "***********************************************************************************************"
        )
        print("Optimization Finished!")
        print(
            "Total time:{:.4f}s".format(time.time() - t_total),
            "Best validation accuracy:{:.4f}".format(self.best_acc_val),
            "EM iterations:{:04d}\n".format(iterations),
        )

    def train_base_model(self, features, adj, labels, mask_train, mask_val, iter):
        best_acc_val = 0
        optimizer = optim.Adam(
            self.base_model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

        t = time.time()
        for epoch in range(self.config["epoch"]):
            self.base_model.train()
            optimizer.zero_grad()

            hidden_output, output = self.base_model(features, adj, return_hidden=True)
            loss_train = F.nll_loss(output[mask_train], labels[mask_train])
            acc_train = accuracy(output[mask_train], labels[mask_train])
            loss_train.backward()
            optimizer.step()

            # Evaluate valset performance (deactivate dropout)
            self.base_model.eval()
            hidden_output, output = self.base_model(features, adj, return_hidden=True)

            loss_val = F.nll_loss(output[mask_val], labels[mask_val])
            acc_val = accuracy(output[mask_val], labels[mask_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                if acc_val > self.best_acc_val:
                    self.best_acc_val = acc_val
                    self.best_graph = adj
                    self.hidden_output = hidden_output
                    self.output = output
                    self.weights = deepcopy(self.base_model.state_dict())
                    if self.config["debug"]:
                        print(
                            "=== Saving current graph/base_model, best_acc_val:%.4f"
                            % self.best_acc_val.item()
                        )

            if self.config["debug"]:
                if epoch % 1 == 0:
                    print(
                        "Epoch {:04d}".format(epoch + 1),
                        "loss_train:{:.4f}".format(loss_train.item()),
                        "acc_train:{:.4f}".format(acc_train.item()),
                        "loss_val:{:.4f}".format(loss_val.item()),
                        "acc_val:{:.4f}".format(acc_val.item()),
                        "time:{:.4f}s".format(time.time() - t),
                    )

        print(
            "Iteration {:04d}".format(iter),
            "acc_val:{:.4f}".format(best_acc_val.item()),
        )

    def test(self, features, labels, mask_test, hetero=False):
        """Evaluate the performance on testset."""
        print("=== Testing ===")
        print("Picking the best model according to validation performance")

        self.base_model.load_state_dict(self.weights)
        self.base_model.eval()
        hidden_output, output = self.base_model(
            features, self.best_graph, return_hidden=True
        )
        loss_test = F.nll_loss(output[mask_test], labels[mask_test])
        if hetero:
            pred = torch.argmax(output, dim=1)
            return torch_f1_score(pred[mask_test], labels[mask_test])
        else:
            acc_test = accuracy(output[mask_test], labels[mask_test])
            print(
                "Testset results: ",
                "loss={:.4f}".format(loss_test.item()),
                "accuracy={:.4f}".format(acc_test.item()),
            )
            return acc_test.item()


class EstimateAdj:
    def __init__(
        self, features, labels, adj, idx_train, num_class, num_node, homophily
    ):
        self.num_class = num_class
        self.num_node = num_node
        self.idx_train = idx_train.cpu().numpy()
        self.label = labels.cpu().numpy()
        self.adj = adj.cpu().numpy()

        self.output = None
        self.iterations = 0

        self.homophily = homophily

    def reset_obs(self):
        self.N = 0
        self.E = np.zeros((self.num_node, self.num_node), dtype=np.int64)

    def update_obs(self, output):
        self.E += output
        self.N += 1

    def revise_pred(self):
        for j in range(len(self.idx_train)):
            self.output[self.idx_train[j]] = self.label[self.idx_train[j]]

    def E_step(self, Q):
        """Run the Expectation(E) step of the EM algorithm.
        Parameters
        ----------
        Q:
            The current estimation that each edge is actually present (numpy.array)

        Returns
        ----------
        alpha:
            The estimation of true-positive rate (float)
        beta:
            The estimation of false-positive rate (float)
        O:
            The estimation of network model parameters (numpy.array)
        """
        # Temporary variables to hold the numerators and denominators of alpha and beta
        an = Q * self.E
        an = np.triu(an, 1).sum()
        bn = (1 - Q) * self.E
        bn = np.triu(bn, 1).sum()
        ad = Q * self.N
        ad = np.triu(ad, 1).sum()
        bd = (1 - Q) * self.N
        bd = np.triu(bd, 1).sum()

        # Calculate alpha, beta
        alpha = an * 1.0 / (ad)
        beta = bn * 1.0 / (bd)

        O = np.zeros((self.num_class, self.num_class))

        n = []
        counter = Counter(self.output)
        for i in range(self.num_class):
            n.append(counter[i])

        a = self.output.repeat(self.num_node).reshape(self.num_node, -1)
        for j in range(self.num_class):
            c = a == j
            for i in range(j + 1):
                b = a == i
                O[i, j] = np.triu((b & c.T) * Q, 1).sum()
                if i == j:
                    O[j, j] = 2.0 / (n[j] * (n[j] - 1)) * O[j, j]
                else:
                    O[i, j] = 1.0 / (n[i] * n[j]) * O[i, j]
        return (alpha, beta, O)

    def M_step(self, alpha, beta, O):
        """Run the Maximization(M) step of the EM algorithm."""
        O += O.T - np.diag(O.diagonal())

        row = self.output.repeat(self.num_node)
        col = np.tile(self.output, self.num_node)
        tmp = O[row, col].reshape(self.num_node, -1)

        p1 = tmp * np.power(alpha, self.E) * np.power(1 - alpha, self.N - self.E)
        p2 = (1 - tmp) * np.power(beta, self.E) * np.power(1 - beta, self.N - self.E)
        Q = p1 * 1.0 / (p1 + p2 * 1.0)
        return Q

    def EM(self, output, tolerance=0.000001):
        """Run the complete EM algorithm.
        Parameters
        ----------
        tolerance:
            Determine the tolerance in the variantions of alpha, beta and O, which is acceptable to stop iterating (float)
        seed:
            seed for np.random.seed (int)

        Returns
        ----------
        iterations:
            The number of iterations to achieve the tolerance on the parameters (int)
        """
        # Record previous values to confirm convergence
        alpha_p = 0
        beta_p = 0

        self.output = output
        self.revise_pred()

        # Do an initial E-step with random alpha, beta and O
        # Beta must be smaller than alpha
        beta, alpha = np.sort(np.random.rand(2))
        O = np.triu(np.random.rand(self.num_class, self.num_class))

        # Calculate initial Q
        Q = self.M_step(alpha, beta, O)

        while abs(alpha_p - alpha) > tolerance or abs(beta_p - beta) > tolerance:
            alpha_p = alpha
            beta_p = beta
            alpha, beta, O = self.E_step(Q)
            Q = self.M_step(alpha, beta, O)
            self.iterations += 1

        if self.homophily > 0.5:
            Q += self.adj
        return (alpha, beta, O, Q, self.iterations)
