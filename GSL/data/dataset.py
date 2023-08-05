import os.path as osp
import pickle as pkl
import platform
import sys
import urllib.request
import zipfile
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import dgl
import time
import json
from dgl import ops
from sklearn.model_selection import train_test_split
from GSL.utils import (sample_mask, sparse_mx_to_torch_sparse_tensor, to_scipy, to_tensor,
                       dense_adj_to_edge_index, row_normalize_features, to_undirected)
from GSL.metric import InnerProductSimilarity, CosineSimilarity
from GSL.processor import KNNSparsify, kneighbors_graph, KNearestNeighbour


def get_mask(idx, labels):
    mask = np.zeros(labels.shape[0], dtype=np.bool)
    mask[idx] = 1
    return mask

def get_y(idx, labels):
    mx = np.zeros(labels.shape)
    mx[idx] = labels[idx]
    return mx


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

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Normalize sparse adjacency matrix"""
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix: symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def random_coauthor_amazon_splits(labels, lcc_mask=None):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    num_classes = labels.max().item() + 1
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (labels[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (labels == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    # train_mask = index_to_mask(train_index, size=data.num_nodes)
    # val_mask = index_to_mask(val_index, size=data.num_nodes)
    # test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return train_index, val_index, rest_index


class Dataset:
    """
    Dataset class contains:
        three citation network datasets: "cora", "citeseer" and "pubmed";
        two product network datasets: "amazon-computers" and "amazon-photo";
        two ogb benchmark datasets: "ogbn-arxiv" and "ogbn-products"
        five heterophilous datasets: "roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"
    The 'cora', 'citeseer' and 'pubmed' are downloaded from https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/
    The 'amazon-computers' and 'amazon-photo' are downloaded from https://github.com/shchur/gnn-benchmark/raw/master/data/npz/
    The 'ogbn-arxiv' and 'ogbn-products' are downloaded from http://snap.stanford.edu/ogb/data/nodeproppred
    The 'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers' and 'questions' are downloaded from https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/master/data/

    Note that 'minesweeper', 'tolokers' and 'questions' need to be evaluated by ROC AUC metric.

    Parameters
    ----------
    root : string
        root directory where the dataset should be saved.
    name : string
        dataset name, it can be chosen from ['cora', 'citeseer', 'pubmed',
        'amazon-computers', 'amazon-photo',
        'ogbn-arxiv', 'ogbn-products',
        'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']
    seed : int
        random seed for splitting training/validation/test.
    split_num : int
        training/validation/test splits of heterophilic datasets, it can be chosen from [0, 9]


    Examples
    -------- homophilic dataset --------
        >>> from GSL.dataset import Dataset
        >>> data = Dataset(root='/tmp/', name='cora')
        >>> adj, features, labels = data.adj, data.features, data.labels
        >>> train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    -------- heterophilic dataset --------
        >>> from GSL.dataset import Dataset
        >>> data = Dataset(root='/tmp/', name='roman-empire', split_num=0)
        >>> adj, features, labels = data.adj, data.features, data.labels
        >>> train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    -------- Randomly delete edges --------
        >>> from GSL.dataset import Dataset
        >>> data = Dataset(root='/tmp/', name='cora', drop_rate=0.5)
    -------- Use KNN graph --------
        >>> from GSL.dataset import Dataset
        >>> data = Dataset(root='/tmp/', name='cora', k=5, use_knn=True)
    -------- Load mettack graph --------
        >>> from GSL.dataset import Dataset
        >>> data = Dataset(root='/tmp/', name='cora', setting='nettack', ptb_rate=0.05)
    """

    def __init__(self, root, name, seed=None, use_mettack=False, ptb_rate=0):
        self.name = name.lower()

        assert self.name in [
            "cora",
            "citeseer",
            "pubmed",
            "ogbn-arxiv",
            "cornell",
            "wisconsin",
            "texas",
            "actor"
        ], (
            "Currently only support cora, citeseer, pubmed, "
            + "ogbn-arxiv, "
            + "cornell, wisconsin, texas, actor"
        )

        self.seed = seed

        if platform.system() == "Windows":
            self.root = root
        else:
            self.root = osp.expanduser(osp.normpath(root))

            if not use_mettack:
                adj, self.features, self.labels = self.load_data()
            else:
                if self.name in ['cora', 'citeseer', 'polblogs']:
                    if not osp.exists(osp.join(self.root, '{}_features.npz'.format(self.name))):
                        url = 'https://raw.githubusercontent.com/likuanppd/STABLE/master/ptb_graphs/'
                        self.download('{}_features.npz'.format(self.name), url)
                    if not osp.exists(osp.join(self.root, '{}_labels.npy'.format(self.name))):
                        url = 'https://raw.githubusercontent.com/likuanppd/STABLE/master/ptb_graphs/'
                        self.download('{}_labels.npy'.format(self.name), url)
                    features = sp.load_npz(osp.join(self.root, '{}_features.npz'.format(self.name)))
                    self.labels = torch.LongTensor(np.load(osp.join(self.root, '{}_labels.npy'.format(self.name))))

                    idx_train = np.load(osp.join(self.root, 'mettack_{}_{}_idx_train.npy'.format(self.name, ptb_rate)))
                    idx_val = np.load(osp.join(self.root, 'mettack_{}_{}_idx_val.npy'.format(self.name, ptb_rate)))
                    idx_test = np.load(osp.join(self.root, 'mettack_{}_{}_idx_test.npy'.format(self.name, ptb_rate)))
                    adj = torch.load(osp.join(self.root, 'mettack_{}_{}.pt'.format(self.name, ptb_rate)))
                    adj = to_scipy(adj)
                    _, features = to_tensor(adj, features)
                    self.features = features.to_dense()
                    self.get_mask(self.labels, idx_train, idx_val, idx_test)
            
            self.adj = adj

    def to(self, device):
        self.adj = self.adj.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)

    def load_npz(self, file_name, is_sparse=True):
        with np.load(file_name) as loader:
            # loader = dict(loader)
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                            loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                 loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                if 'attr_data' in loader:
                    features = loader['attr_data']
                else:
                    features = None
                labels = loader.get('labels')
        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def get_adj(self, data_filename):
        adj, features, labels = self.load_npz(data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        lcc = self.largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels = labels[lcc]
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        features = torch.FloatTensor(features.todense())
        labels = torch.LongTensor(labels)

        return adj, features, labels
    
    def largest_connected_components(self, adj, n_components=1):
        """Select k largest connected components.

		Parameters
		----------
		adj : scipy.sparse.csr_matrix
			input adjacency matrix
		n_components : int
			n largest connected components we want to select
		"""

        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print("Selecting {0} largest connected components".format(n_components))
        return nodes_to_keep

    def load_data(self):
        print("Loading {} dataset...".format(self.name))
        if self.name in ['cora', 'citeseer', 'pubmed']:
            return self.load_citation_dataset()

        if self.name in ['ogbn-arxiv']:
            return self.load_ogb()
        
        if self.name in ['cornell', 'wisconsin', 'texas', 'actor']:
            return self.load_heterophilous()
        
    @property
    def num_feat(self):
        return self.features.shape[1]
    
    @property
    def num_class(self):
        return self.labels.max().item() + 1
    
    @property
    def num_nodes(self):
        return self.features.shape[0]
    
    @property
    def num_edges(self):
        return int(np.sum(self.adj) / 2)
    
    @property
    def edge_homophily(self):
        src = sp.coo_matrix(self.adj).row
        dst = sp.coo_matrix(self.adj).col
        homophily_ratio = 1.0 * torch.sum((self.labels[src] == self.labels[dst])) / src.shape[0]
        return homophily_ratio.item()
    
    @property
    def avg_degree(self):
        return np.mean(np.sum(self.adj, axis=1))
    
    def to(self, device):
        self.features = self.features.to(device)
        self.adj = self.adj.to(device)
        self.labels = self.labels.to(device)
        try:
            self.train_mask = self.train_mask.to(device)
            self.val_mask = self.val_mask.to(device)
            self.test_mask = self.test_mask.to(device)
        except:
            self.train_masks = [self.train_masks[i].to(device) for i in range(10)]
            self.val_masks = [self.val_masks[i].to(device) for i in range(10)]
            self.test_masks = [self.test_masks[i].to(device) for i in range(10)]
        return self

    def download(self, name, url):
        try:
            print("Downloading", osp.join(url, name))
            urllib.request.urlretrieve(url + name, osp.join(self.root, name))
            print("Done!")
        except:
            raise Exception(
                """Download failed! Make sure you have stable Internet connection and enter the right name"""
            )

    def load_citation_dataset(self):
        url = "https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/"
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            name = "ind.{}.{}".format(self.name, names[i])
            data_filename = osp.join(self.root, name)

            if not osp.exists(data_filename):
                self.download(name, url)

            with open(data_filename, "rb") as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding="latin1"))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_file = "ind.{}.test.index".format(self.name)
        if not osp.exists(osp.join(self.root, test_idx_file)):
            self.download(test_idx_file, url)

        def parse_index_file(filename):
            index = []
            for line in open(filename):
                index.append(int(line.strip()))
            return index

        test_idx_reorder = parse_index_file(osp.join(self.root, test_idx_file))
        test_idx_range = np.sort(test_idx_reorder)

        if self.name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)
        self.idx_test = torch.LongTensor(idx_test)
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        features = torch.FloatTensor(features.todense())
        labels = torch.LongTensor(labels)
        self.train_mask = torch.BoolTensor(train_mask)
        self.val_mask = torch.BoolTensor(val_mask)
        self.test_mask = torch.BoolTensor(test_mask)

        for i in range(labels.shape[0]):
            sum_ = torch.sum(labels[i])
            if sum_ != 1:
                labels[i] = torch.tensor([1, 0, 0, 0, 0, 0])
        labels = (labels == 1).nonzero()[:, 1]

        return adj, features, labels

    def load_ogb(self):
        from ogb.nodeproppred import NodePropPredDataset

        dataset = NodePropPredDataset(name=self.name, root=self.root)

        split_idx = dataset.get_idx_split()

        data = dataset[0] # This dataset has only one graph
        features = torch.Tensor(data[0]['node_feat'])
        labels = torch.LongTensor(data[1]).squeeze(-1)

        edge_index = data[0]['edge_index']
        adj = to_undirected(edge_index, num_nodes=data[0]['num_nodes'])
        assert adj.diagonal().sum() == 0 and adj.max() <= 1 and (adj != adj.transpose()).sum() == 0

        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        self.get_mask(labels, train_idx, valid_idx, test_idx)

        return adj, features, labels
    
    def load_heterophilous(self):
        url = "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/master/data/"

        name = f'{self.name}.npz'
        data_filename = osp.join(self.root, name)
        if not osp.exists(data_filename):
            self.download(name, url)

        data = np.load(data_filename)
        features = torch.FloatTensor(data['node_features'])
        labels = torch.LongTensor(data['node_labels'])
        edges = torch.tensor(data['edges'])
        row, col = edges[:, 0], edges[:, 1]
        size = edges.max().item() + 1
        adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size))
        adj = nx.adjacency_matrix(nx.from_numpy_array(adj.todense()))

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])
        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        l = self.onehot(labels)
        self.train_masks = [torch.BoolTensor(get_mask(train_idx, labels)) for train_idx in train_idx_list]
        self.val_masks = [torch.BoolTensor(get_mask(val_idx, labels)) for val_idx in val_idx_list]
        self.test_masks = [torch.BoolTensor(get_mask(test_idx, labels)) for test_idx in test_idx_list]
        return adj, features, labels

    def __repr__(self):
        return "{0}(adj_shape={1}, feature_shape={2})".format(
            self.name, self.adj.shape, self.features.shape
        )


    def get_mask(self, labels, train_idx, val_idx, test_idx):

        labels = self.onehot(labels)

        self.train_mask = torch.BoolTensor(get_mask(train_idx))
        self.val_mask = torch.BoolTensor(get_mask(val_idx))
        self.test_mask = torch.BoolTensor(get_mask(test_idx))
        self.y_train, self.y_val, self.y_test = (
            get_y(train_idx),
            get_y(val_idx),
            get_y(test_idx),
        )

    def onehot(self, labels):
        eye = np.identity(labels.max() + 1)
        onehot_mx = eye[labels]
        return onehot_mx


if __name__ == "__main__":
    from GSL.data import Dataset

    # Citation dataset
    data_path = osp.join(osp.expanduser('~'), 'datasets')
    data = Dataset(root=data_path, name="cora", seed=0)
    adj, features, labels = data.adj, data.features, data.labels
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
