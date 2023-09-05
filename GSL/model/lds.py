from GSL.model import BaseModel
from GSL.encoder import GCN_dgl, GCN, MetaDenseGCN
from GSL.utils import (
    empirical_mean_loss,
    Metrics,
    accuracy,
    get_triu_values,
    graph_regularization,
    get_lr,
    is_square_matrix,
    lds_normalize_adjacency_matrix,
    shuffle_splits_,
    split_mask,
    straight_through_estimator,
    lds_to_undirected,
    triu_values_to_symmetric_matrix,
    row_normalize_features,
    normalize,
    dense_adj_to_edge_index,
    to_undirected,
)


from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple, Optional, List, Union
from copy import deepcopy
from collections import OrderedDict
from higher.optim import DifferentiableOptimizer, DifferentiableAdam


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging
import copy


from torch import Tensor
from torch.nn import Parameter
from torch.distributions import Bernoulli
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import StepLR


def setup_basic_logger():
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s (%(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger


class Transform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data):
        pass


class ShuffleSplits(Transform):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed

    def __call__(self, data):
        assert (
            hasattr(data, "train_mask")
            and hasattr(data, "val_mask")
            and hasattr(data, "test_mask")
        )
        copy_data = copy.deepcopy(data)
        shuffle_splits_(copy_data, seed=self.seed)
        return copy_data


class LDS(BaseModel):
    """
    LDS model for node classification.
    """

    def __init__(
        self, num_features, num_classes, metric, config_path, dataset_name, device, data
    ):
        super(LDS, self).__init__(
            num_features, num_classes, metric, config_path, dataset_name, device
        )

        # inner_trainer里自己有一个gcn
        # outer_trainer里自己有一个graph_generator

        self.logger = setup_basic_logger()

        # self.num_features = num_features
        # self.graph_nhid = int(self.config.hid_graph.split(":")[0])
        # self.graph_nhid2 = int(self.config.hid_graph.split(":")[1])
        # self.nhid = self.config.nhid
        # self.conv1 = GCNConv(num_features, self.nhid)
        # self.conv2 = GCNConv(self.nhid, num_classes)

        # if self.config.sparse:
        #     self.conv_graph = GCNConv_diag(num_features, device)
        #     self.conv_graph2 = GCNConv_diag(num_features, device)
        # else:
        #     self.conv_graph = GCNConv(num_features, self.graph_nhid)
        #     self.conv_graph2 = GCNConv(self.graph_nhid, self.graph_nhid2)

        # self.F = ({'relu': F.relu, 'prelu': F.prelu, 'tanh': torch.tanh})[self.config.F]
        # self.F_graph = ({'relu': F.relu, 'prelu': F.prelu, 'tanh': torch.tanh})[self.config.F_graph]
        # self.dropout = self.config.dropout
        # self.K = self.config.compl_param.split(":")[0]
        # self.mask = None
        # self.Adj_new = None
        # self._normalize = self.config.normalize
        # self.reduce = self.config.reduce
        # self.sparse = self.config.sparse
        # self.norm_mode = "sym"
        # self.config = self.config

    def fit(self, dataset=None, split_num=0):
        adj, features, labels = (
            dataset.adj.clone(),
            dataset.features.clone(),
            dataset.labels,
        )
        features = row_normalize_features(features)

        dataset.features = features
        dataset.adj = adj

        if dataset.name in ["cornell", "texas", "wisconsin", "actor"]:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
            dataset.train_mask, dataset.val_mask, dataset.test_mask = (
                train_mask,
                val_mask,
                test_mask,
            )
        # else:
        #     shuffler = ShuffleSplits(self.config.seed)
        #     dataset = shuffler(dataset)

        # from IPython import embed; embed(header="in LDS fit 1")

        dataset = copy.deepcopy(dataset)
        # prevent the original dataset from being modified
        dataset.val_mask, outer_opt_mask = split_mask(
            dataset.val_mask, ratio=0.5, shuffle=True, device=self.device
        )

        # from IPython import embed; embed(header="in LDS fit 2")

        # from IPython import embed; embed(header='in scripts bilevel')
        # Normal GCN
        # graph_convolutional_network = GCN(self.num_feat, self.config.hidden_size, self.num_class, num_layers=2,
        #                                   dropout=self.config.dropout, dropout_adj=0.0, sparse=False,
        #                                   conv_bias=False, activation_last=self.config.activation_last)
        # TorchMeta GCN
        graph_convolutional_network = MetaDenseGCN(
            self.num_feat,
            self.config.hidden_size,
            self.num_class,
            self.config.dropout,
            normalize_adj=self.config.normalize_adj,
        ).to(device=self.device)
        self.inner_trainer = InnerProblemTrainer(
            model=graph_convolutional_network,
            data=dataset,
            lr=self.config.gcn_optimizer_learning_rate,
            weight_decay=self.config.gcn_weight_decay,
        )

        graph_model_factory = GraphGenerativeModelFactory(data=dataset)
        graph_generator_model = graph_model_factory.create(self.config.graph_model).to(
            self.device
        )
        graph_generator_opt = graph_model_factory.optimizer(graph_generator_model)
        self.outer_trainer = OuterProblemTrainer(
            optimizer=graph_generator_opt,
            data=dataset,
            opt_mask=outer_opt_mask,
            model=graph_generator_model,
        )

        patience = self.config.patience
        outer_loop_max_epochs = self.config.outer_loop_max_epochs
        inner_loop_max_epochs = self.config.inner_loop_max_epochs
        hyper_gradient_interval = self.config.hyper_gradient_interval

        ### BilevelProblemRunner
        outer_early_stopper = EarlyStopping(
            patience=patience, max_epochs=outer_loop_max_epochs
        )
        current_step = 0
        outer_step = 0

        best_val_result = 0

        while (
            not outer_early_stopper.abort
        ):  # Depends on empirical mean validation loss
            inner_early_stopper = EarlyStopping(
                patience=patience, max_epochs=inner_loop_max_epochs
            )

            self.inner_trainer.reset_weights()
            self.inner_trainer.reset_optimizer()

            # TODO
            # self.logger.info("Starting new outer loop...")

            while not inner_early_stopper.abort:
                train_set_metrics = self.inner_opt_step()
                inner_early_stopper.update(
                    train_set_metrics.loss,
                    model_params=self.inner_trainer.copy_model_params(),
                )
                # TODO
                # self.logger.info(
                #     f"Model Optimization Step {current_step}: "
                #     f"loss={train_set_metrics.loss}, accuracy={train_set_metrics.acc}"
                # )

                """
                Optimize Graph Parameters every 'hyper_gradient_interval' steps
                """
                if (
                    hyper_gradient_interval == 0
                    or current_step % hyper_gradient_interval == 0
                ):
                    self.hyper_opt_step(current_step)

                current_step += 1

            # TODO
            # self.logger.info(f"Exited inner optimization")

            gcn_model_params = inner_early_stopper.model_params

            self.outer_trainer.train(False)
            empirical_val_results, empirical_test_results = empirical_mean_loss(
                self.inner_trainer.model,
                graph_model=self.outer_trainer.model,
                n_samples=self.config.n_samples_empirical_mean,
                data=dataset,  # TODO: data/dataset??? which one
                model_parameters=gcn_model_params,
            )

            self.logger.info(
                f"Empirical Validation Set Results: loss={empirical_val_results.loss}, "
                f"accuracy={empirical_val_results.acc}"
            )
            self.logger.info(
                f"Empirical Test Set Results: loss={empirical_test_results.loss}, "
                f"accuracy={empirical_test_results.acc}"
            )

            if empirical_val_results.acc > best_val_result:
                best_val_result = empirical_val_results.acc
                self.best_result = empirical_test_results.acc

            # from IPython import embed; embed(header="in bilevel_trainer after test")
            outer_early_stopper.update(
                empirical_val_results.loss,
                model_params=[
                    deepcopy(gcn_model_params),
                    self.outer_trainer.model.state_dict(),
                ],
            )
            outer_step += 1

        self.logger.info(f"Ended training after {outer_step} steps...")
        self.gcn_params, self.graph_state_dict = outer_early_stopper.model_params
        
        evaluate_result = self.evaluate(dataset)

        if evaluate_result['acc.val.final'] > best_val_result:
            best_val_result = evaluate_result['acc.val.final']
            self.best_result = evaluate_result['acc.test.final']

        # 原作者早停的逻辑是val.loss小于patience个loss平均值时加入，是一个更鲁棒的算法，但和常规的update best result不一样，这里还是采用常规的update best result
        # 使用acc来取最好的一次，而不是loss


    def inner_opt_step(self):
        self.outer_trainer.train()
        graph = self.outer_trainer.sample()
        # from IPython import embed; embed(header="inner_opt_step")
        train_set_metrics = self.inner_trainer.train_step(graph)
        return train_set_metrics

    def hyper_opt_step(self, current_step: int):
        # print(f"Optimizing graph parameters at step {current_step}")
        # TODO
        # self.logger.info(f"Optimizing graph parameters at step {current_step}")
        metrics = self.outer_trainer.train_step(self.inner_trainer.model_forward)
        # 通过self.inner_trainer.model_forward传入outer_trainer.train_step，
        # 这样会导致self.inner_trainer.model有梯度，但我之后并不需要这个梯度了，
        # 因此使用self.inner_trainer.detach()来截断梯度
        self.inner_trainer.detach()
        self.outer_trainer.detach()

        # if sacred_runner is not None:
        #     sacred_runner.log_scalar("loss.outer", metrics.loss, step=current_step)
        #     sacred_runner.log_scalar("acc.outer", metrics.acc, step=current_step)
        #     for i, lr in enumerate(self.outer_trainer.get_learning_rates()):
        #         sacred_runner.log_scalar(f"Outer Learning Rate {i}", lr, step=current_step)

        # TODO
        # self.logger.info(f"Graph Model Statistics:")
        # for name, value in self.outer_trainer.model.statistics().items():
        #     self.logger.info(f"{name}: {value}")
        # self.logger.info(
        #     f"Performance on held-out sample for graph optimization: loss={metrics.loss}, accuracy={metrics.acc} "
        #     f"Outer optimizer learning rate: {self.outer_trainer.get_learning_rates()}"
        # )

    def evaluate(self, dataset):
        assert (
            self.gcn_params is not None and self.graph_state_dict is not None
        ), "Models need to be trained before evaluation."
        model_params, graph_model_state_dict = self.gcn_params, self.graph_state_dict
        self.outer_trainer.model.load_state_dict(graph_model_state_dict)

        empirical_val_results, empirical_test_results = empirical_mean_loss(
            self.inner_trainer.model,
            graph_model=self.outer_trainer.model,
            n_samples=self.config.n_samples_empirical_mean,
            data=dataset,
            model_parameters=model_params,
        )
        return {
            "loss.val.final": empirical_val_results.loss,
            "acc.val.final": empirical_val_results.acc,
            "loss.test.final": empirical_test_results.loss,
            "acc.test.final": empirical_test_results.acc,
        }


class InnerProblemTrainer:
    def __init__(
        self,
        model,
        data,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_params = OrderedDict(model.named_parameters())
        self.optimizer: DifferentiableOptimizer = None
        self.data = data

        self.reset_optimizer()

    def reset_weights(self):
        # MetaTorchGCN
        # self.model.reset_parameters()
        self.model.reset_weights()
        self.model_params = OrderedDict(self.model.named_parameters())

    def reset_optimizer(self) -> None:
        optimizer = Adam(
            # MetaTorchGCN
            # [
            #     {"params": self.model.layers[0].parameters(), "weight_decay": self.weight_decay},
            #     {"params": self.model.layers[1].parameters()}
            # ], lr=self.lr)
            [
                {
                    "params": self.model.layer_in.parameters(),
                    "weight_decay": self.weight_decay,
                },
                {"params": self.model.layer_out.parameters()},
            ],
            lr=self.lr,
        )
        self.optimizer = DifferentiableAdam(optimizer, self.model.parameters())

    def copy_detach_parameter_dict(self, parameters):
        _params_dict = parameters.copy()
        for key in _params_dict.keys():
            _params_dict[key] = _params_dict[key].detach().clone().requires_grad_(True)
        return _params_dict

    def copy_model_params(self):
        return self.copy_detach_parameter_dict(self.model_params)

    def train_step(self, graph: torch.Tensor, mask: torch.Tensor = None) -> Metrics:
        """
        Does one training step with differentiable optimizer
        :param graph: Sampled graph as adjacency matrix
        :param mask: Optional, use mask other than training set
        :return: Training loss and accuracy
        """
        assert is_square_matrix(graph)

        predictions = self.model_forward(graph, is_train=True)
        mask = mask or self.data.train_mask
        loss = F.nll_loss(predictions[mask], self.data.labels[mask])
        acc = accuracy(predictions[mask], self.data.labels[mask])

        # from IPython import embed; embed(header="in inner train_step")
        new_model_params = self.optimizer.step(loss, params=self.model_params.values())
        self._update_model_params(list(new_model_params))

        return Metrics(loss=loss.item(), acc=acc)

    def model_forward(self, graph, is_train=True):
        self.model.train(mode=is_train)
        # return self.model(self.data.features, graph, params=self.model_params) # TODO: figure out the torchmeta's influence
        # it seems to be adding bias?
        # graph = lds_normalize_adjacency_matrix(graph)
        # MetaTorchGCN
        return self.model(self.data.features, graph, params=self.model_params)
        # from IPython import embed; embed()
        # return self.model(self.data.features, graph)

    def evaluate(self, graph: torch.Tensor, mask: torch.Tensor = None) -> Metrics:
        """
        Calculate validation set metrics
        :param graph: Graph as adjacency matrix
        :param mask: Optional, specify mask other than validation set
        :return: Wrapper class containing loss and accuracy
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model_forward(graph, is_train=False)
            mask = mask or self.data.val_mask
            loss = F.nll_loss(predictions[mask], self.data.labels[mask])
            acc = accuracy(predictions[mask], self.data.labels[mask])

        return Metrics(loss=loss.item(), acc=acc)

    def detach(self):
        """
        Detach and overwrite model parameters in place to stop gradient flow. Allows for truncated backpropagation
        """
        self.model_params = self.copy_detach_parameter_dict(self.model_params)

        self.detach_optimizer()

    def _update_model_params(self, new_model_params: List[torch.Tensor]):
        for parameter_index, parameter_name in enumerate(self.model_params.keys()):
            self.model_params[parameter_name] = new_model_params[parameter_index]

    def detach_optimizer(self):
        """Removes all params from their compute graph in place."""
        # detach param groups
        for group in self.optimizer.param_groups:
            for k, v in group.items():
                if isinstance(v, torch.Tensor):
                    v.detach_().requires_grad_()

        # detach state
        for state_dict in self.optimizer.state:
            for k, v_dict in state_dict.items():
                if isinstance(k, torch.Tensor):
                    k.detach_().requires_grad_()
                for k2, v2 in v_dict.items():
                    if isinstance(v2, torch.Tensor):
                        v2.detach_().requires_grad_()


class OuterProblemTrainer:
    def __init__(
        self,
        optimizer,
        data,
        opt_mask: Tensor,
        model,
        smoothness_factor: float = 0.0,
        disconnection_factor: float = 0.0,
        sparsity_factor: float = 0.0,
        regularize: float = False,
        lr_decay: float = 1.0,
        lr_decay_step_size: int = 1,
        refine_embeddings: bool = False,
        pretrain: bool = False,
    ):
        self.lr_decay = lr_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.dataset = data
        self.opt_mask = opt_mask
        self.model = model

        self.regularize = regularize
        self.smoothness_factor = smoothness_factor
        self.disconnection_factor = disconnection_factor
        self.sparsity_factor = sparsity_factor

        self.optimizer = optimizer
        self.lr_decayer = (
            StepLR(
                self.optimizer, step_size=self.lr_decay_step_size, gamma=self.lr_decay
            )
            if self.lr_decay is not None
            else None
        )

        self.refine_embeddings = refine_embeddings

        if pretrain:
            self.pretrain_model()

    def train_step(self, gcn_predict_fct, mask=None, retain_graph=True):
        self.model.train()
        self.optimizer.zero_grad()
        graph = self.model.sample()
        predictions = gcn_predict_fct(graph)
        mask = mask or self.opt_mask
        loss = F.nll_loss(predictions[mask], self.dataset.labels[mask])
        acc = accuracy(predictions[mask], self.dataset.labels[mask])

        if self.regularize:
            loss += graph_regularization(
                graph=graph,
                features=self.dataset.features,
                smoothness_factor=self.smoothness_factor,
                disconnection_factor=self.disconnection_factor,
                sparsity_factor=self.sparsity_factor,
            )

        # from IPython import embed; embed(header="in outer train_step")

        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

        if self.lr_decayer is not None:
            self.lr_decayer.step(epoch=None)

        self.model.project_parameters()

        if self.refine_embeddings:
            self.model.refine()
        return Metrics(loss=loss.item(), acc=acc)

    def sample(self) -> Tensor:
        return self.model.sample()

    def detach(self):
        self.model.load_state_dict(self.model.state_dict())
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def get_learning_rates(self) -> List[float]:
        if self.optimizer is None:
            raise ValueError(
                "Can't get optimizer learning rate, no optimizer initialized yet."
            )
        return get_lr(self.optimizer)

    def train(self, mode=True):
        self.model.train(mode=mode)

    def eval(self):
        self.model.eval()

    # def pretrain_model(self) -> None:
    #     pretrainer = PretrainerFactory.trainer(model=self.model, data=self.dataset)
    #     pretrainer.train()


class EarlyStopping:
    def __init__(self, patience: int, max_epochs: int = 10000):
        self.abort = False
        self.patience = patience
        self.model_state_dict = None
        self.model_params = None
        self.max_epochs = max_epochs
        self.curr_step = 0

        self.losses = list()

    def update(
        self,
        new_value,
        model: torch.nn.Module = None,
        model_params: Union[Dict, torch.Tensor, List] = None,
    ):
        self.losses.append(new_value)

        if self.curr_step <= self.patience or new_value <= np.mean(
            self.losses[-(self.patience + 1) : -1]
        ):
            if model is not None:
                self.model_state_dict = model.state_dict()
            if model_params is not None:
                self.model_params = model_params
        else:
            self.abort = True
        if self.curr_step is not None and self.curr_step >= self.max_epochs:
            self.abort = True

        self.curr_step = self.curr_step + 1

    def best_model_state_dict(self):
        return self.model_state_dict


class SPARSIFICATION(Enum):
    NONE = 1
    KNN = 2
    EPS = 3


def knn_graph_dense(
    x: torch.Tensor, k: int, loop: bool = True, metric: str = "cosine"
) -> torch.Tensor:
    from sklearn.neighbors import kneighbors_graph

    graph = kneighbors_graph(
        x.numpy(), n_neighbors=k, mode="connectivity", metric=metric, include_self=loop
    )
    return torch.FloatTensor(graph.toarray())


def sparsify(
    edge_probs: Tensor,
    sparsification: SPARSIFICATION,
    embeddings: Optional[Tensor] = None,
    k: Optional[int] = None,
    eps: Optional[float] = None,
    knn_metric: str = "cosine",
) -> Tensor:
    if sparsification == SPARSIFICATION.NONE:
        return edge_probs
    elif sparsification == SPARSIFICATION.KNN:
        assert embeddings is not None, "Needs embeddings to create knn graph"
        assert k is not None and 0 < k < edge_probs.size(0)
        if knn_metric == "dot":
            knn_metric = np.dot
        knn_graph = knn_graph_dense(
            embeddings.detach().cpu(), k=k, loop=False, metric=knn_metric
        ).to(edge_probs.device)
        edges_not_in_knn_graph = (knn_graph == 0.0).nonzero(as_tuple=True)
        edge_probs = edge_probs.clone()
        edge_probs[
            edges_not_in_knn_graph
        ] = 0.0  # Stops gradient flow through these edges!
        return edge_probs
    elif sparsification == SPARSIFICATION.EPS:
        assert eps is not None
        allowed_edge_indices = (edge_probs < eps).nonzero(as_tuple=True)
        edge_probs = edge_probs.clone()
        edge_probs[
            allowed_edge_indices
        ] = 0.0  # Stops gradient flow through these edges!
        return edge_probs
    else:
        raise NotImplementedError()


def sample_graph(
    edge_probs: Tensor,
    undirected: bool,
    embeddings: Optional[Tensor] = None,
    dense: bool = False,
    k: Optional[int] = 20,
    sparsification: SPARSIFICATION = SPARSIFICATION.NONE,
    force_straight_through_estimator: bool = False,
    eps: Optional[float] = 0.9,
    knn_metric: str = "cosine",
) -> Tensor:
    assert is_square_matrix(edge_probs)
    assert embeddings is None or edge_probs.size(0) == embeddings.size(0)

    if dense:
        sample = sparsify(
            edge_probs,
            sparsification=sparsification,
            embeddings=embeddings,
            k=k,
            eps=eps,
            knn_metric=knn_metric,
        )
    else:
        bernoulli_sample = Bernoulli(probs=edge_probs).sample()
        sample = sparsify(
            bernoulli_sample,
            sparsification=sparsification,
            embeddings=embeddings,
            k=k,
            eps=eps,
            knn_metric=knn_metric,
        )

    sample = lds_to_undirected(sample, from_triu_only=True) if undirected else sample
    if force_straight_through_estimator or not dense:
        sample = straight_through_estimator(sample, edge_probs)
    return sample


class Sampler:
    def config():
        undirected: bool = True
        k: int = 20
        eps: float = 0.9
        sparsification: str = "NONE"
        dense: bool = False
        knn_metric: str = "cosine"

    def sample(
        edge_probs: Tensor,
        undirected: bool = True,
        sparsification: str = "NONE",
        k: int = 20,
        eps: float = 0.9,
        embeddings: Tensor = None,
        dense: bool = False,
        knn_metric: str = "cosine",
    ) -> Tensor:
        """
        Gets square matrix with bernoulli parameters of graph distribution.
        Uses straight-through estimator to use gradient information even for not-sampled edges
        :param embeddings: When available pass raw embeddings to enable knn-graph construction
        :param k: Number of nearest neighbors to use
        :param sparsification: How to sparsify the graph. Leave empty or specify method (only 'KNN',
        'EPS' or  'NONE' supported)
        :param edge_probs: Bernoulli Parameters. Needs to be a square matrix
        :param undirected: Only use bernoulli parameters from triu matrix and ignore others

        :return:
        """
        assert sparsification in SPARSIFICATION.__members__
        sparsification_method = SPARSIFICATION[sparsification]
        sampled_graph = sample_graph(
            edge_probs=edge_probs,
            embeddings=embeddings,
            undirected=undirected,
            sparsification=sparsification_method,
            dense=dense,
            k=k,
            eps=eps,
            knn_metric=knn_metric,
        )
        return sampled_graph


class GraphGenerativeModel(nn.Module, ABC):
    def __init__(self, sample_undirected: bool = True, *args, **kwargs):
        super(GraphGenerativeModel, self).__init__(*args, **kwargs)
        self.sample_undirected = sample_undirected

    def sample(self, *args, **kwargs) -> Tensor:
        probs = self.forward()
        # from IPython import embed; embed(header='GraphGenerativeModel.sample')
        edges = Sampler.sample(probs)
        return edges

    def project_parameters(self):
        pass

    def refine(self):
        pass

    @abstractmethod
    def statistics(self) -> Dict[str, float]:
        pass


class GraphGenerativeModelFactory:
    def __init__(self, data):
        self.data = data

    def create(self, model_name: str):
        if model_name == "lds":
            model = self.lds(data=self.data)
        else:
            raise NotImplementedError(f"Model {model_name} not supported.")
        return model  # type: ignore

    def optimizer(self, model: GraphGenerativeModel) -> Optimizer:
        model_type = type(model)
        if model_type == BernoulliGraphModel:
            opt = self.lds_optimizer(model=model)
        else:
            raise NotImplementedError(
                f"Optimizer for model type {model_type} not implemented."
            )
        return opt  # type: ignore

    #############
    # LDS MODEL #
    #############
    def _lds_config():
        directed: bool = False
        lr: float = 1.0

    def lds(self, data, directed: bool = False):
        return BernoulliGraphModel(data.adj, directed=directed)

    def lds_optimizer(self, model, lr: float = 1.0) -> Optimizer:
        optimizer = SGD(model.parameters(), lr=lr)
        return optimizer

    @staticmethod
    def get_optimizer(optimizer_type):
        if optimizer_type.lower() == "sgd":
            return SGD
        elif optimizer_type.lower() == "adam":
            return Adam
        else:
            raise NotImplementedError()


class ParameterClamper(object):
    def __call__(self, module):
        for param in module.parameters():
            w = param.data
            w.clamp_(0.0, 1.0)


class BernoulliGraphModel(GraphGenerativeModel):
    def __init__(self, init_matrix: Tensor, directed: bool = False):
        """
        :param directed:
        :param init_matrix: Either symmetric matrix or flattened
            array of the values of the upper triangular matrix
        """
        super(BernoulliGraphModel, self).__init__()
        assert is_square_matrix(init_matrix)

        self.directed = directed
        self.orig_matrix = init_matrix

        # Init Values
        probs = init_matrix if directed else get_triu_values(init_matrix)
        self.probs = Parameter(probs, requires_grad=True)

    def project_parameters(self):
        self.apply(ParameterClamper())

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # from IPython import embed; embed(header='BernoulliGraphModel.forward')
        return self.probs if self.directed else triu_values_to_symmetric_matrix(self.probs)  # type: ignore

    def statistics(self) -> Dict[str, float]:
        sample = self.forward()
        n_edges = sample.size(0) ** 2
        return {
            "expected_num_edges": sample.sum().item(),
            "percentage_edges_expected": sample.sum().item() / n_edges,
            "mean_prob": torch.mean(self.probs).item(),
            "min_prob": torch.min(self.probs).item(),
            "max_prob": torch.max(self.probs).item(),
        }
