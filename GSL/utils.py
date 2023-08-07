import datetime
import json
import os
import pickle
import random
import shutil
import time
from copy import deepcopy

import dgl
import easydict
import networkx as nx
import numpy as np
import pandas as pd
import pytz
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dgl.data import DGLDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from GSL.metric import CosineSimilarity
from GSL.processor import KNearestNeighbour

EOS = 1e-10
VERY_SMALL_NUMBER = 1e-12


def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
        configs = [easydict.EasyDict(yaml.safe_load(raw_text))]
    return configs


def save_config(cfg, path):
    with open(os.path.join(path, "config.yaml"), "w") as fo:
        yaml.dump(dict(cfg), fo)


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(
        feat_node, size=int(feat_node * mask_rate), replace=False
    )
    mask[:, samples] = 1
    return mask.cuda(), samples


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0, :], indices[1, :]
    dgl_graph = dgl.graph(
        (rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device="cuda"
    )
    dgl_graph.edata["w"] = values.detach().cuda()
    return dgl_graph


def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1.0 / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1.0 / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1.0 / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = (
                inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
            )

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1.0 / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def accuracy(output, labels):
    if not hasattr(labels, "__len__"):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def auc_f1_mima(logits, label):
    preds = torch.argmax(logits, dim=1)
    test_f1_macro = f1_score(label.cpu(), preds.cpu(), average="macro")
    test_f1_micro = f1_score(label.cpu(), preds.cpu(), average="micro")

    best_proba = F.softmax(logits, dim=1)
    if logits.shape[1] != 2:
        auc = roc_auc_score(
            y_true=label.detach().cpu().numpy(),
            y_score=best_proba.detach().cpu().numpy(),
            multi_class="ovr",
        )
    else:
        auc = roc_auc_score(
            y_true=label.detach().cpu().numpy(),
            y_score=best_proba[:, 1].detach().cpu().numpy(),
        )
    return test_f1_macro, test_f1_micro, auc


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index : index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1) : (end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1) : (end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1) : (end) * (k + 1)] = (
            torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        )
        norm_row[index:end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5)
    return rows, cols, values


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.0

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata["w"].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx


def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    assert stratify is not None, "stratify cannot be None!"

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(
        idx,
        random_state=None,
        train_size=train_size + val_size,
        test_size=test_size,
        stratify=stratify,
    )

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(
        idx_train_and_val,
        random_state=None,
        train_size=(train_size / (train_size + val_size)),
        test_size=(val_size / (train_size + val_size)),
        stratify=stratify,
    )

    return idx_train, idx_val, idx_test


def get_train_val_test_gcn(labels, seed=None):
    """This setting follows gcn, where we randomly sample 20 instances for each class
    as training data, 500 instances as validation data, 1000 instances as test data.
    Note here we are not using fixed splits. When random seed changes, the splits
    will also change.

    Parameters
    ----------
    labels : numpy.array
        node labels
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels == i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[:20])).astype(np.int32)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[20:])).astype(np.int32)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[:500]
    idx_test = idx_unlabeled[500:1500]
    return idx_train, idx_val, idx_test


def get_random_mask(features, r, nr):
    nones = torch.sum(features > 0.0).float()
    nzeros = features.shape[0] * features.shape[1] - nones
    pzeros = nones / nzeros / r * nr
    probs = torch.zeros(features.shape).cuda()
    probs[features == 0.0] = pzeros
    probs[features > 0.0] = 1 / r
    mask = torch.bernoulli(probs)
    return mask


def get_random_mask_ogb(features, r):
    probs = torch.full(features.shape, 1 / r)
    mask = torch.bernoulli(probs)
    return mask


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.history = []
        self.last = None
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.last = self.mean()
        self.history.append(self.last)
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix(
            (values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape
        )
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix(
            (values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape
        )


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def to_tensor(adj, features, labels=None, device="cpu"):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor on target device.
    Args:
        adj : scipy.sparse.csr_matrix
            the adjacency matrix.
        features : scipy.sparse.csr_matrix
            node features
        labels : numpy.array
            node labels
        device : str
            'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)


def sparse_mx_to_sparse_tensor(sparse_mx):
    """sparse matrix to sparse tensor matrix(torch)
    Args:
        sparse_mx : scipy.sparse.csr_matrix
            sparse matrix
    """
    sparse_mx_coo = sparse_mx.tocoo().astype(np.float32)
    sparse_row = torch.LongTensor(sparse_mx_coo.row).unsqueeze(1)
    sparse_col = torch.LongTensor(sparse_mx_coo.col).unsqueeze(1)
    sparse_indices = torch.cat((sparse_row, sparse_col), 1)
    sparse_data = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(
        sparse_indices.t(), sparse_data, torch.Size(sparse_mx.shape)
    )


class EarlyStopping:
    def __init__(self, patience=10, path=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.best_weight = None
        self.best_loss = None
        self.path = path

    def step(self, acc, model, epoch):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_weight = deepcopy(model.state_dict())
        elif score < self.best_score:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter}/{self.patience}, best_val_score:{self.best_score:.4f} at E{self.best_epoch}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_weight = deepcopy(model.state_dict())
            self.counter = 0

        return self.early_stop

    def loss_step(self, loss, model, epoch):
        """

        Parameters
        ----------
        loss Float or torch.Tensor

        model torch.nn.Module

        Returns
        -------

        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if self.best_loss is None:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_weight = deepcopy(model.state_dict())
        elif loss >= self.best_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, best_val_loss:{self.best_loss:.4f} at E{self.best_epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if loss < self.best_loss:
                self.best_weight = deepcopy(model.state_dict())
                self.best_epoch = epoch
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

def true_positive(pred, target, n_class):
    return torch.tensor([((pred == i) & (target == i)).sum() for i in range(n_class)])


def false_positive(pred, target, n_class):
    return torch.tensor([((pred == i) & (target != i)).sum() for i in range(n_class)])


def false_negative(pred, target, n_class):
    return torch.tensor([((pred != i) & (target == i)).sum() for i in range(n_class)])


def precision(tp, fp):
    res = tp / (tp + fp)
    res[torch.isnan(res)] = 0
    return res


def recall(tp, fn):
    res = tp / (tp + fn)
    res[torch.isnan(res)] = 0
    return res


def f1_score(prec, rec):
    f1_score = 2 * (prec * rec) / (prec + rec)
    f1_score[torch.isnan(f1_score)] = 0
    return f1_score


def cal_maf1(tp, fp, fn):
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    ma_f1 = f1_score(prec, rec)
    return torch.mean(ma_f1).cpu().numpy()


def cal_mif1(tp, fp, fn):
    gl_tp, gl_fp, gl_fn = torch.sum(tp), torch.sum(fp), torch.sum(fn)
    gl_prec = precision(gl_tp, gl_fp)
    gl_rec = recall(gl_tp, gl_fn)
    mi_f1 = f1_score(gl_prec, gl_rec)
    return mi_f1.cpu().numpy()


def macro_f1(pred, target, n_class):
    tp = true_positive(pred, target, n_class).to(torch.float)
    fn = false_negative(pred, target, n_class).to(torch.float)
    fp = false_positive(pred, target, n_class).to(torch.float)

    ma_f1 = cal_maf1(tp, fp, fn)
    return ma_f1


def micro_f1(pred, target, n_class):
    tp = true_positive(pred, target, n_class).to(torch.float)
    fn = false_negative(pred, target, n_class).to(torch.float)
    fp = false_positive(pred, target, n_class).to(torch.float)

    mi_f1 = cal_mif1(tp, fp, fn)
    return mi_f1


def sparse_dense_mul(s, d):
    if not s.is_sparse:
        return s * d
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())


def prob_to_adj(mx, threshold):
    mx = np.triu(mx, 1)
    mx += mx.T
    (row, col) = np.where(mx > threshold)
    adj = sp.coo_matrix(
        (np.ones(row.shape[0]), (row, col)),
        shape=(mx.shape[0], mx.shape[0]),
        dtype=np.int64,
    )
    adj = sparse_mx_to_sparse_tensor(adj)
    return adj


def get_homophily(label, adj):
    label = label.cpu().numpy()
    adj = adj.cpu().numpy()
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = np.triu((label == label.T) & (adj == 1)).sum(axis=0)
    d = np.triu(adj).sum(axis=0)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1.0 / d[i])
    return np.mean(homos)


def diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    diff_ = diff_ / torch.clamp(norm_, min=VERY_SMALL_NUMBER)
    return diff_


def SquaredFrobeniusNorm(X):
    return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def row_normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    if isinstance(features, torch.Tensor):
        rowsum = torch.sum(features, dim=1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        features = r_mat_inv @ features
    else:
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    return features


def mx_normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    

def dense_adj_to_edge_index(adj):
    edge_index = sp.coo_matrix(adj.cpu())
    indices = np.vstack((edge_index.row, edge_index.col))
    edge_index = torch.LongTensor(indices).to(adj.device)
    return edge_index


def k_fold(dataset, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    if isinstance(dataset, DGLDataset):
        labels = dataset.graph_labels
        for _, idx in skf.split(torch.zeros(len(dataset)), labels):
            test_indices.append(torch.from_numpy(idx).to(torch.long))
    elif isinstance(dataset, list):
        for _, idx in skf.split(
            torch.zeros(len(dataset)), [data.y for data in dataset]
        ):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def transform_relation_graph_list(hg, category, identity=True):

    # get target category id
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    g = dgl.to_homogeneous(hg, ndata="h")
    # find out the target node ids in g
    loc = (g.ndata[dgl.NTYPE] == category_id).to("cpu")
    category_idx = torch.arange(g.num_nodes())[loc]

    edges = g.edges()
    etype = g.edata[dgl.ETYPE]
    ctx = g.device
    # g.edata['w'] = torch.ones(g.num_edges(), device=ctx)
    num_edge_type = torch.max(etype).item()

    # norm = EdgeWeightNorm(norm='right')
    # edata = norm(g.add_self_loop(), torch.ones(g.num_edges() + g.num_nodes(), device=ctx))
    graph_list = []
    for i in range(num_edge_type + 1):
        e_ids = torch.nonzero(etype == i).squeeze(-1)
        sg = dgl.graph((edges[0][e_ids], edges[1][e_ids]), num_nodes=g.num_nodes())
        # sg.edata['w'] = edata[e_ids]
        sg.edata["w"] = torch.ones(sg.num_edges(), device=ctx)
        graph_list.append(sg)
    if identity == True:
        x = torch.arange(0, g.num_nodes(), device=ctx)
        sg = dgl.graph((x, x))
        # sg.edata['w'] = edata[g.num_edges():]
        sg.edata["w"] = torch.ones(g.num_nodes(), device=ctx)
        graph_list.append(sg)
    return graph_list, g.ndata["h"], category_idx


def get_nodes_dict(hg):
    n_dict = {}
    for n in hg.ntypes:
        n_dict[n] = hg.num_nodes(n)
    return n_dict


def to_undirected(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    else:
        num_nodes = max(num_nodes, edge_index.max() + 1)

    row, col = edge_index
    data = np.ones(edge_index.shape[1])
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = (adj + adj.transpose()) > 0
    return adj.astype(np.float64)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def random_drop_edge(adj, drop_rate):
    row, col = adj.nonzero()
    num_nodes = max(row.max(), col.max()) + 1
    edge_num = adj.nnz
    drop_edge_num = int(edge_num * drop_rate)
    edge_mask = np.ones(edge_num, dtype=np.bool)
    indices = np.random.permutation(edge_num)[:drop_edge_num]
    edge_mask[indices] = False
    row, col = row[edge_mask], col[edge_mask]
    data = np.ones(edge_num - drop_edge_num)
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj


def random_add_edge(adj, add_rate):
    row, col = adj.nonzero()
    num_nodes = max(row.max(), col.max()) + 1
    edge_num = adj.nnz
    num_edges_to_add = int(edge_num * add_rate)
    row_ = np.random.randint(0, num_nodes, size=(num_edges_to_add,))
    col_ = np.random.randint(0, num_nodes, size=(num_edges_to_add,))
    new_row = np.concatenate((row, row_), axis=0)
    new_col = np.concatenate((col, col_), axis=0)
    data = np.ones(edge_num + num_edges_to_add)
    adj = sp.csr_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
    return adj


def get_knn_graph(features, k, dataset):
    metric = CosineSimilarity()
    adj = metric(features, features)
    adj = KNearestNeighbour(k=k)(adj).numpy()
    if dataset != "ogbn-arxiv":
        adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
    else:
        row, col = adj.nonzero()
        num_nodes = max(max(row), max(col)) + 1
        edge_index = np.array([row, col])
        adj = to_undirected(edge_index, num_nodes)
    return adj


def feature_mask(features, missing_rate):
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask


def apply_feature_mask(features, mask):
    features[mask] = float(0)

def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)

def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f"Start running {func.__name__} at {get_cur_time()}")
        ret = func(*args, **kw)
        print(
            f"Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}."
        )
        return ret

    return wrapper


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)

def graph_edge_to_lot(g):
    # graph_edge_to list of (row_id, col_id) tuple
    return list(
        map(tuple, np.column_stack([_.cpu().numpy() for _ in g.edges()]).tolist())
    )


def min_max_scaling(input, type="col"):
    """
    min-max scaling modified from https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122/5

    Parameters
    ----------
    input (2 dimensional torch tensor): input data to scale
    type (str): type of scaling, row, col, or global.

    Returns (2 dimensional torch tensor): min-max scaled torch tensor
    -------
    Example input tensor (list format):
        [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    Scaled tensor (list format):
        [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0]]

    """
    if type in ["row", "col"]:
        dim = 0 if type == "col" else 1
        input -= input.min(dim).values
        input /= input.max(dim).values
        # corner case: the row/col's minimum value equals the maximum value.
        input[input.isnan()] = 0
        return input
    elif type == "global":
        return (input - input.min()) / (input.max() - input.min())
    else:
        ValueError("Invalid type of min-max scaling.")


def edge_lists_to_set(_):
    return set(list(map(tuple, _)))


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:

        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def mkdir_list(p_list, use_relative_path=True, log=True):
    """Create directories for the specified path lists.
        Parameters
        ----------
        p_list :Path lists

    """
    # ! Note that the paths MUST END WITH '/' !!!
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    for p in p_list:
        p = os.path.join(root_path, p) if use_relative_path else p
        p = os.path.dirname(p)
        mkdir_p(p, log)



def save_pickle(var, f_name):
    mkdir_list([f_name])
    pickle.dump(var, open(f_name, "wb"))
    print(f"File {f_name} successfully saved!")


def load_pickle(f_name):
    return pickle.load(open(f_name, "rb"))


from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.0:
            raise ValueError("max_decay_steps should be greater than 1.")
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [
            (base_lr - self.end_learning_rate)
            * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [
                (base_lr - self.end_learning_rate)
                * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
                + self.end_learning_rate
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group["lr"] = lr


import math


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, inputSize, K, T=0.07, device=None):
        super(MemoryMoCo, self).__init__()
        self.device = device
        self.queueSize = K
        self.T = T
        self.index = 0

        self.register_buffer("params", torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory", torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        print("using queue shape: ({},{})".format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()
        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)
        out = torch.cat((l_pos, l_neg), dim=1)

        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).to(self.device)
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self, device):
        super(NCESoftmaxLoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz], device=self.device).long()
        loss = self.criterion(x, label)
        return loss


def moment_update(model, model_ema, m):
    """model_ema = m * model_ema + (1 - m) model"""
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def para_copy(model_to_init, pretrained_model, paras_to_copy):
    # Pass parameters (if exists) of old model to new model
    para_dict_to_update = model_to_init.gnn.state_dict()
    pretrained_dict = {
        k: v for k, v in pretrained_model.state_dict().items() if k in paras_to_copy
    }
    para_dict_to_update.update(pretrained_dict)
    model_to_init.gnn.load_state_dict(para_dict_to_update)


def global_topk(input, k, largest):
    # https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    v, i = torch.topk(input.flatten(), k, largest=largest)
    return np.array(np.unravel_index(i.cpu().numpy(), input.shape)).T.tolist()