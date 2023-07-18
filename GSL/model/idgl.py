import gc
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from GSL.encoder import GAT, GCN, GraphSAGE
from GSL.metric import *
from GSL.model import BaseModel
from GSL.processor import Normalize, ThresholdSparsify
from GSL.utils import (AverageMeter, DummyLogger, SquaredFrobeniusNorm,
                       accuracy, diff, row_normalize_features, to_scipy)

VERY_SMALL_NUMBER = 1e-12
_SAVED_WEIGHTS_FILE = "params.saved"


def normalize_adj(mx):
    """Normalize sparse adjacency matrix"""
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


class IDGL(BaseModel):
    """
    Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings (NeurIPS 2020')
    """

    def __init__(
        self, num_features, num_classes, metric, config_path, dataset_name, device
    ):
        super(IDGL, self).__init__(
            num_features, num_classes, metric, config_path, dataset_name, device
        )
        self.config.update({"num_feat": num_features, "num_class": num_classes})
        self.model = GraphClf(self.config, device).to(device)
        self.config = self.config
        self._train_metrics = {"nloss": AverageMeter(), "acc": AverageMeter()}
        self._dev_metrics = {"nloss": AverageMeter(), "acc": AverageMeter()}
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        self.criterion = F.nll_loss
        self.score_func = accuracy
        self.metric_name = "acc"
        self._init_optimizer()
        self.net_module = GraphClf
        # self.dirname = self.logger.dirname
        seed = self.config.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device != "cpu":
            torch.cuda.manual_seed(seed)
        self.is_test = False

    def fit(self, dataset, split_num=0):
        adj, features, labels = (
            dataset.adj.clone(),
            dataset.features.clone(),
            dataset.labels,
        )
        if dataset.name in ["cornell", "texas", "wisconsin", "actor"]:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = (
                dataset.train_mask,
                dataset.val_mask,
                dataset.test_mask,
            )

        if adj.is_sparse:
            indices = adj.coalesce().indices()
            values = adj.coalesce().values()
            shape = adj.coalesce().shape
            num_nodes = features.shape[0]
            loop_edge_index = torch.stack(
                [torch.arange(num_nodes), torch.arange(num_nodes)]
            ).to(adj.device)
            loop_edge_index = torch.cat([indices, loop_edge_index], dim=1)
            loop_values = torch.ones(num_nodes).to(adj.device)
            loop_values = torch.cat([values, loop_values], dim=0)
            adj = torch.sparse_coo_tensor(
                indices=loop_edge_index, values=loop_values, size=shape
            )
        else:
            adj += torch.eye(adj.shape[0]).to(self.device)
            adj = adj.to_sparse()
        idx_train, idx_val, idx_test = (
            torch.nonzero(train_mask).squeeze(),
            torch.nonzero(val_mask).squeeze(),
            torch.nonzero(test_mask).squeeze(),
        )
        adj = to_scipy(adj)
        features = row_normalize_features(features)
        adj = normalize_adj(adj)
        adj = torch.from_numpy(adj.todense()).to(torch.float)
        self.is_test = False
        self._epoch = self._best_epoch = 0
        self.train_loader = {
            "adj": adj.to(self.device),
            "features": features.to(self.device),
            "labels": labels.to(self.device),
            "idx_train": idx_train,
            "idx_val": idx_val,
        }
        self.dev_loader = self.train_loader
        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = -float("inf")
        self._reset_metrics()

        while self._stop_condition(self._epoch, self.config["patience"]):
            self._epoch += 1
            self.run_epoch(
                self.train_loader,
                training=True,
            )
            dev_output, dev_gold = self.run_epoch(
                self.dev_loader,
                training=False,
            )
            cur_dev_score = self._dev_metrics[self.config["eary_stop_metric"]].mean()

            print(
                f"Epoch: {self._epoch: 02d}, "
                f"Loss: {self._train_loss.mean():.4f}, "
                f"Train: {100 * self._train_metrics['acc'].mean():.2f}%, "
                f"Valid: {100 * self._dev_metrics['acc'].mean():.2f}%, "
            )

            if self._best_metrics[self.config["eary_stop_metric"]] < cur_dev_score:
                self._best_epoch = self._epoch
                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k].mean()
                self.best_model = self.model

            self._reset_metrics()
        # format_str = self.summary()
        # print(format_str)
        # self.logger.write_to_file(format_str)
        self.best_result = self.test(idx_test, hetero=False)["acc"].item()

    def test(self, idx_test, hetero=False):
        self.train_loader.update({"idx_test": idx_test})
        self._n_test_examples = idx_test.shape[0]
        self.test_loader = self.train_loader
        # print("Restoring best model")
        # self.init_saved_network(self.dirname)
        # self.model = self.model.to(self.device)
        self.model = self.best_model
        self.is_test = True
        self._reset_metrics()
        for param in self.model.parameters():
            param.requires_grad = False

        output, gold = self.run_epoch(
            self.test_loader,
            training=False,
        )

        metrics = self._dev_metrics
        format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
            self._n_test_examples, 1, 1
        )
        format_str += self.metric_to_str(metrics)
        if hetero:
            # pred = torch.argmax(output, dim=1)
            # ma_f1, mi_f1 = torch_f1_score(pred, gold, self.config.num_class)
            # format_str += (
            #     f"\nFinal score on the testing set: {ma_f1:0.5f} {mi_f1:0.5f}\n"
            # )
            pass
        else:
            test_score = self.score_func(output, gold)
            format_str += "\nFinal score on the testing set: {:0.5f}\n".format(
                test_score
            )
        # print(format_str)
        # self.logger.write_to_file(format_str)
        # self.logger.close()

        test_metrics = {}
        for k in metrics:
            test_metrics[k] = metrics[k].mean()
        # if hetero:
            # return ma_f1, mi_f1
        if test_score is not None:
            test_metrics[self.metric_name] = test_score
        return test_metrics

    def run_epoch(self, data_loader, training=True):
        """BP after all iterations"""
        mode = "train" if training else ("test" if self.is_test else "dev")
        self.model.train(training)

        init_adj, features, labels = (
            data_loader["adj"],
            data_loader["features"],
            data_loader["labels"],
        )

        if mode == "train":
            idx = data_loader["idx_train"]
        elif mode == "dev":
            idx = data_loader["idx_val"]
        else:
            idx = data_loader["idx_test"]

        network = self.model
        # Init
        features = F.dropout(
            features,
            network.config.get("feat_adj_dropout", 0),
            training=network.training,
        )
        init_node_vec = features

        cur_raw_adj, cur_adj = network.learn_graph(
            network.metric1,
            network.processors,
            init_node_vec,
            network.graph_skip_conn,
            graph_include_self=network.graph_include_self,
            init_adj=init_adj,
        )
        if self.config["graph_learn"] and self.config.get("max_iter", 10) > 0:
            cur_raw_adj = F.dropout(
                cur_raw_adj,
                network.config.get("feat_adj_dropout", 0),
                training=network.training,
            )
        cur_adj = F.dropout(
            cur_adj,
            network.config.get("feat_adj_dropout", 0),
            training=network.training,
        )

        node_vec = torch.relu(network.encoder.layers[0](init_node_vec, cur_adj))
        node_vec = F.dropout(node_vec, network.dropout, training=network.training)

        # Add mid GNN layers
        for encoder in network.encoder.layers[1:-1]:
            node_vec = torch.relu(encoder(node_vec, cur_adj))
            node_vec = F.dropout(node_vec, network.dropout, training=network.training)

        # BP to update weights
        output = network.encoder.layers[-1](node_vec, cur_adj)
        output = F.log_softmax(output, dim=-1)

        score = self.score_func(output[idx], labels[idx])
        loss1 = self.criterion(output[idx], labels[idx])

        if self.config["graph_learn"] and self.config["graph_learn_regularization"]:
            loss1 += self.add_graph_loss(cur_raw_adj, init_node_vec)

        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        if not mode == "test":
            if self._epoch > self.config.get("pretrain_epoch", 0):
                max_iter_ = self.config.get("max_iter", 10)  # Fine-tuning
                if self._epoch == self.config.get("pretrain_epoch", 0) + 1:
                    for k in self._dev_metrics:
                        self._best_metrics[k] = -float("inf")

            else:
                max_iter_ = 0  # Pretraining
        else:
            max_iter_ = self.config.get("max_iter", 10)

        if training:
            eps_adj = float(
                self.config.get("eps_adj", 0)
            )  # cora: 5.5e-8, cora w/o input graph: 1e-8, citeseer w/o input graph: 1e-8, wine: 2e-5, cancer: 2e-5, digtis: 2e-5
        else:
            eps_adj = float(
                self.config.get("test_eps_adj", self.config.get("eps_adj", 0))
            )

        pre_raw_adj = cur_raw_adj

        loss = 0
        iter_ = 0
        while (
            self.config["graph_learn"]
            and (
                iter_ == 0
                or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj
            )
            and iter_ < max_iter_
        ):
            iter_ += 1
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = network.learn_graph(
                network.metric2,
                network.processors,
                node_vec,
                network.graph_skip_conn,
                graph_include_self=network.graph_include_self,
                init_adj=init_adj,
            )

            update_adj_ratio = self.config.get("update_adj_ratio", None)
            if update_adj_ratio is not None:
                cur_adj = (
                    update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj
                )

            node_vec = torch.relu(network.encoder.layers[0](init_node_vec, cur_adj))
            node_vec = F.dropout(
                node_vec, self.config.get("gl_dropout", 0), training=network.training
            )

            for encoder in network.encoder.layers[1:-1]:
                node_vec = torch.relu(encoder(node_vec, cur_adj))
                node_vec = F.dropout(
                    node_vec,
                    self.config.get("gl_dropout", 0),
                    training=network.training,
                )

            output = network.encoder.layers[-1](node_vec, cur_adj)
            output = F.log_softmax(output, dim=-1)
            score = self.score_func(output[idx], labels[idx])
            loss += self.criterion(output[idx], labels[idx])

            if self.config["graph_learn"] and self.config["graph_learn_regularization"]:
                loss += self.add_graph_loss(cur_raw_adj, init_node_vec)

            if self.config["graph_learn"] and not self.config.get(
                "graph_learn_ratio", None
            ) in (None, 0):
                loss += SquaredFrobeniusNorm(
                    cur_raw_adj - pre_raw_adj
                ) * self.config.get("graph_learn_ratio")

        # if mode == "test" and self.config.get("out_raw_learned_adj_path", None):
        #     out_raw_learned_adj_path = os.path.join(
        #         self.dirname, self.config["out_raw_learned_adj_path"]
        #     )
        #     np.save(out_raw_learned_adj_path, cur_raw_adj.cpu())
        #     print("Saved raw_learned_adj to {}".format(out_raw_learned_adj_path))

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1

        # del cur_raw_adj, cur_adj, init_adj, first_raw_adj, first_adj, init_node_vec, pre_raw_adj
        # gc.collect()
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.clip_grad()
            self.optimizer.step()

        self._update_metrics(
            loss.item(),
            {"nloss": -loss.item(), self.metric_name: score},
            1,
            training=training,
        )
        return output[idx], labels[idx]

    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config["max_epochs"]
        return False if exceeded_max_epochs or no_improvement else True

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def _init_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                parameters,
                self.config["learning_rate"],
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                parameters,
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "adamax":
            self.optimizer = optim.Adamax(parameters, lr=self.config["learning_rate"])
        else:
            raise RuntimeError("Unsupported optimizer: %s" % self.config["optimizer"])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config["lr_reduce_factor"],
            patience=self.config["lr_patience"],
            verbose=True,
        )

    def init_saved_network(self, save_dir):
        # Load all saved fields.
        fname = os.path.join(save_dir, _SAVED_WEIGHTS_FILE)
        print("[ Loading saved model %s ]" % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params["state_dict"]
        self.network = self.net_module(self.config, self.device)

        # Merge the arguments
        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict["network"].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    def add_graph_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += (
            self.config["smoothness_ratio"]
            * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features)))
            / int(np.prod(out_adj.shape))
        )
        ones_vec = torch.ones(out_adj.size(-1), device=self.device)
        graph_loss += (
            -self.config["degree_ratio"]
            * torch.mm(
                ones_vec.unsqueeze(0),
                torch.log(
                    torch.mm(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER
                ),
            ).squeeze()
            / out_adj.shape[-1]
        )
        graph_loss += (
            self.config["sparsity_ratio"]
            * torch.sum(torch.pow(out_adj, 2))
            / int(np.prod(out_adj.shape))
        )
        return graph_loss

    def clip_grad(self):
        # Clip gradients
        if self.config["grad_clipping"]:
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, self.config["grad_clipping"])

    def metric_to_str(self, metrics):
        format_str = ""
        for k in metrics:
            format_str += " | {} = {:0.5f}".format(k.upper(), metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = "\n"
        for k in metrics:
            format_str += "{} = {:0.5f}\n".format(k.upper(), metrics[k])
        return format_str

    def save(self, dirname):
        params = {
            "state_dict": {
                "network": self.model.state_dict(),
            },
            "config": self.config,
            "dir": dirname,
        }
        try:
            torch.save(params, os.path.join(dirname, _SAVED_WEIGHTS_FILE))
        except BaseException:
            print("[ WARN: Saving failed... continuing anyway. ]")

    def summary(self):
        start = "\n<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}; ".format(self._best_epoch) + self.best_metric_to_str(
            self._best_metrics
        )
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if k not in metrics:
                    continue
                self._train_metrics[k].update(metrics[k], batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if k not in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k], batch_size)


class GraphClf(nn.Module):
    def __init__(self, config, device):
        super(GraphClf, self).__init__()
        self.config = config
        self.name = "GraphClf"
        self.graph_learn = config["graph_learn"]
        self.graph_metric_type = config["graph_metric_type"]
        self.graph_module = config["graph_module"]
        self.device = device
        nfeat = config["num_feat"]
        nclass = config["num_class"]
        hidden_size = config["hidden_size"]
        self.dropout = config["dropout"]
        self.graph_skip_conn = config["graph_skip_conn"]
        self.graph_include_self = config.get("graph_include_self", True)
        self.scalable_run = config.get("scalable_run", False)

        if self.graph_module == "gcn":
            gcn_module = GCN
            self.encoder = gcn_module(
                in_channels=nfeat,
                hidden_channels=hidden_size,
                out_channels=nclass,
                num_layers=config.get("graph_hops", 2),
                dropout=self.dropout,
                dropout_adj=0,
                sparse=config.get("sparse", False),
            )

        elif self.graph_module == "gat":
            self.encoder = GAT(
                nfeat=nfeat,
                nhid=hidden_size,
                nclass=nclass,
                dropout=self.dropout,
                nheads=config.get("gat_nhead", 1),
                alpha=config.get("gat_alpha", 0.2),
            )

        elif self.graph_module == "graphsage":
            self.encoder = GraphSAGE(
                nfeat,
                hidden_size,
                nclass,
                1,
                F.relu,
                self.dropout,
                config.get("graphsage_agg_type", "gcn"),
            )

        else:
            raise RuntimeError("Unknown graph_module: {}".format(self.graph_module))

        self.metric1 = WeightedCosine(nfeat, num_pers=config["graph_learn_num_pers"])
        self.metric2 = WeightedCosine(
            hidden_size, num_pers=config["graph_learn_num_pers"]
        )
        self.metric1 = self.metric1.to(device)
        self.metric2 = self.metric2.to(device)
        self.processors = [ThresholdSparsify(config["graph_learn_epsilon"])]

    def learn_graph(
        self,
        metric,
        processors,
        node_features,
        graph_skip_conn=None,
        graph_include_self=False,
        init_adj=None,
    ):
        raw_adj = metric(node_features)
        for processor in processors:
            raw_adj = processor(raw_adj)

        if self.graph_metric_type in ("kernel", "weighted_cosine"):
            assert raw_adj.min().item() >= 0
            adj = raw_adj / torch.clamp(
                torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER
            )

        elif self.graph_metric_type == "cosine":
            adj = (raw_adj > 0).float()
            adj = Normalize()(adj)

        else:
            adj = torch.softmax(raw_adj, dim=-1)

        if graph_skip_conn in (0, None):
            if graph_include_self:
                adj = adj + torch.eye(adj.size(0), device=self.device)
        else:
            adj = (1 - graph_skip_conn) * adj + graph_skip_conn * init_adj

        return raw_adj, adj

    def forward(self, node_features, init_adj=None):
        node_features = F.dropout(
            node_features,
            self.config.get("feat_adj_dropout", 0),
            training=self.training,
        )
        raw_adj, adj = self.learn_graph(
            self.metric1,
            self.processors,
            node_features,
            self.graph_skip_conn,
            init_adj=init_adj,
        )
        adj = F.dropout(
            adj, self.config.get("feat_adj_dropout", 0), training=self.training
        )
        node_vec = self.encoder(node_features, adj)
        output = F.log_softmax(node_vec, dim=-1)
        return output, adj
