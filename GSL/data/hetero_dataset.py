import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import pickle
import scipy
import platform
import torch
import torch.nn.functional as F
import warnings
import dgl
warnings.filterwarnings("ignore",category=DeprecationWarning)

class HeteroDataset():
    """
    Dataset class contains:
        three heterogeneous datasets: "acm", "dblp", and "yelp";
    The 'acm', 'dblp', and 'yelp' are downloaded from https://raw.githubusercontent.com/AndyJZhao/HGSL/master/data.

    Parameters
    ----------
    root : string
        root directory where the dataset should be saved.
    name : string
        dataset name, it can be chosen from ['acm', 'dblp', 'yelp']

    Examples
    --------
	# >>> from GSL.dataset import HeteroDataset
	# >>> data = Dataset(root='/tmp/', name='dblp')
	# >>> adj, features, labels = data.adj, data.features, data.labels
    # >>> train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    # >>> meta_path_emb = data.meta_path_emb
    """

    def __init__(self, root, name, dgl_heterograph=False):
        self.name = name.lower()

        assert self.name in ['acm', 'dblp', 'yelp'], \
                'Currently only support acm, dblp, yelp'

        self.mp_list = {
            'acm': ['psp', 'pap', 'pspap'],
            'dblp': ['apcpa'],
            'yelp': ['bub', 'bsb', 'bublb', 'bubsb']
        }

        self.url = 'https://raw.githubusercontent.com/AndyJZhao/HGSL/master/data/%s/' % self.name

        if platform.system() == 'Windows':
            self.root = root
        else:
            self.root = osp.expanduser(osp.normpath(root))

            self.data_folder = osp.join(root, 'hetero_datasets', self.name) + '/'
            if not osp.exists(self.data_folder):
                os.makedirs(self.data_folder)
            self.feature_filename = self.data_folder + 'node_features.pkl'
            self.edge_filename = self.data_folder + 'edges.pkl'
            self.label_filename = self.data_folder + 'labels.pkl'
            self.metadata_filename = self.data_folder + 'meta_data.pkl'

            self.metadata = {}
            self.mp_emb_dict = {}
            self.features, self.adj, self.meta_path_emb, self.train_idx, self.train_y, \
            self.val_idx, self.val_y, self.test_idx, self.test_y = self.load_data()
            self.metadata['n_meta_path_emb'] = list(self.meta_path_emb.values())[0].shape[1]

        if dgl_heterograph:
            self.construct_dgl_graph()

        if self.name in ['acm', 'dblp', 'yelp']:
            self.merge_labels()
            self.idx2mask()

    @property
    def num_feat(self):
        return self.features.shape[1]
    
    @property
    def num_class(self):
        return self.labels.max().item() + 1

    def construct_dgl_graph(self):
        # transform graph into dgl.heterograph
        if self.name == 'dblp':
            canonical_etypes = [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
                                ('paper', 'paper-conference', 'conference'),
                                ('conference', 'conference-paper', 'paper')]
            target_ntype = 'author'
            meta_paths_dict = {'APCPA': [('author', 'author-paper', 'paper'),
                                         ('paper', 'paper-conference', 'conference'),
                                         ('conference', 'conference-paper', 'paper'),
                                         ('paper', 'paper-author', 'author')],
                               'APA': [('author', 'author-paper', 'paper'),
                                       ('paper', 'paper-author', 'author')],
                               }
        elif self.name == 'acm':
            canonical_etypes = [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
                                ('paper', 'paper-subject', 'subject'), ('subject', 'subject-paper', 'paper')]
            target_ntype = 'paper'
            meta_paths_dict = {'PAPSP': [('paper', 'paper-author', 'author'),
                                         ('author', 'author-paper', 'paper'),
                                         ('paper', 'paper-subject', 'subject'),
                                         ('subject', 'subject-paper', 'paper')],
                               'PAP': [('paper', 'paper-author', 'author'),
                                       ('author', 'author-paper', 'paper')],
                               'PSP': [('paper', 'paper-subject', 'subject'),
                                       ('subject', 'subject-paper', 'paper')]
                               }
        elif self.name == 'yelp':
            canonical_etypes = [('business', 'business-service', 'service'),
                                ('service', 'service-business', 'business'),
                                ('business', 'business-level', 'level'),
                                ('level', 'level-business', 'business'),
                                ('business', 'business-user', 'user'),
                                ('user', 'user-business', 'business')]
            target_ntype = 'business'
            meta_paths_dict = {'BUB': [('business', 'business-user', 'user'),
                                       ('user', 'user-business', 'business')],
                               'BSB': [('business', 'business-service', 'service'),
                                       ('service', 'service-business', 'business')],
                               'BUBLB': [('business', 'business-user', 'user'),
                                         ('user', 'user-business', 'business'),
                                         ('business', 'business-level', 'level'),
                                         ('level', 'level-business', 'business')],
                               'BUBSB': [('business', 'business-user', 'user'),
                                         ('user', 'user-business', 'business'),
                                         ('business', 'business-service', 'service'),
                                         ('service', 'service-business', 'business')]
                               }
        self.canonical_etypes = canonical_etypes
        self.target_ntype = target_ntype
        self.meta_paths_dict = meta_paths_dict

        node_features = self.features
        edges = [v for k, v in self.edges.items()]
        labels = self.labels

        num_nodes = edges[0].shape[0]
        assert len(canonical_etypes) == len(edges)

        ntype_mask = dict()
        ntype_idmap = dict()
        ntypes = set()
        data_dict = {}

        # create dgl graph
        for etype in canonical_etypes:
            ntypes.add(etype[0])
            ntypes.add(etype[2])
        for ntype in ntypes:
            ntype_mask[ntype] = np.zeros(num_nodes, dtype=bool)
            ntype_idmap[ntype] = np.full(num_nodes, -1, dtype=int)
        for i, etype in enumerate(canonical_etypes):
            src_nodes = edges[i].nonzero()[0]
            dst_nodes = edges[i].nonzero()[1]
            src_ntype = etype[0]
            dst_ntype = etype[2]
            ntype_mask[src_ntype][src_nodes] = True
            ntype_mask[dst_ntype][dst_nodes] = True
        for ntype in ntypes:
            ntype_idx = ntype_mask[ntype].nonzero()[0]
            ntype_idmap[ntype][ntype_idx] = np.arange(ntype_idx.size)
        for i, etype in enumerate(canonical_etypes):
            src_nodes = edges[i].nonzero()[0]
            dst_nodes = edges[i].nonzero()[1]
            src_ntype = etype[0]
            dst_ntype = etype[2]
            data_dict[etype] = \
                (torch.from_numpy(ntype_idmap[src_ntype][src_nodes]).type(torch.int64),
                 torch.from_numpy(ntype_idmap[dst_ntype][dst_nodes]).type(torch.int64))
        g = dgl.heterograph(data_dict)

        # split and label
        def idx2mask(idx, len):
            """Create mask."""
            mask = np.zeros(len)
            mask[idx] = 1
            return mask
        all_label = np.full(g.num_nodes(target_ntype), -1, dtype=int)
        for i, split in enumerate(['train', 'val', 'test']):
            node = np.array(labels[i])[:, 0]
            label = np.array(labels[i])[:, 1]
            all_label[node] = label
            g.nodes[target_ntype].data['{}_mask'.format(split)] = \
                torch.from_numpy(idx2mask(node, g.num_nodes(target_ntype))).type(torch.bool)
        g.nodes[target_ntype].data['label'] = torch.from_numpy(all_label).type(torch.long)

        for ntype in ntypes:
            idx = ntype_mask[ntype].nonzero()[0]
            g.nodes[ntype].data['h'] = node_features[idx]

        self.g = g
        self.num_classes = len(torch.unique(self.g.nodes[self.target_ntype].data['label']))
        self.in_dim = self.g.ndata['h'][self.target_ntype].shape[1]

    def idx2mask(self):
        labels = self.onehot(self.labels)

        def get_mask(idx):
            mask = np.zeros(labels.shape[0], dtype=bool)
            mask[idx] = 1
            return mask

        self.train_mask = torch.BoolTensor(get_mask(self.train_idx.cpu()))
        self.val_mask = torch.BoolTensor(get_mask(self.val_idx.cpu()))
        self.test_mask = torch.BoolTensor(get_mask(self.test_idx.cpu()))

    def merge_labels(self):
        labels = torch.zeros(self.train_idx.shape[0] + self.val_idx.shape[0] + self.test_idx.shape[0]).type(torch.LongTensor)
        labels[self.train_idx] = self.train_y
        labels[self.val_idx] = self.val_y
        labels[self.test_idx] = self.test_y
        self.labels = labels

    def load_data(self):
        print('Loading {} dataset...'.format(self.name))

        if self.name in ['acm', 'dblp', 'yelp']:
            if not osp.exists(self.feature_filename):
                self.download_pkl()

        features, adj, meta_path_emb, train_idx, train_y, val_idx, val_y, test_idx, test_y = self.get_adj()
        return features, adj, meta_path_emb, train_idx, train_y, val_idx, val_y, test_idx, test_y

    def download_pkl(self):
        """Download adjacency matrix npz file from self.url.
        """
        print('Downloading from {} to {}'.format(self.url, self.data_folder))
        try:
            for filename in ['node_features.pkl', 'edges.pkl', 'labels.pkl', 'meta_data.pkl']:
                urllib.request.urlretrieve(self.url+filename, self.data_folder+filename)
            for mp in self.mp_list[self.name]:
                mp_file_name = f'{mp}_emb.pkl'
                urllib.request.urlretrieve(self.url+mp_file_name, self.data_folder+mp_file_name)
            print('Done!')
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def get_adj(self):
        with open(self.feature_filename, 'rb') as f:
            self.features = pickle.load(f)
        with open(self.edge_filename, 'rb') as f:
            self.edges = pickle.load(f)
        with open(self.label_filename, 'rb') as f:
            self.labels = pickle.load(f)
        with open(self.metadata_filename, 'rb') as f:
            self.metadata.update(pickle.load(f))
        if scipy.sparse.issparse(self.features):
            self.features = self.features.todense()

        self.load_meta_path_emb()

        features = torch.from_numpy(self.features).type(torch.FloatTensor)
        train_idx, train_y, val_idx, val_y, test_idx, test_y = self.get_label()

        adj = np.sum(list(self.edges.values())).todense()
        adj = torch.from_numpy(adj).type(torch.FloatTensor)
        adj = F.normalize(adj, dim=1, p=2)

        meta_path_emb = {}
        for mp in self.mp_list[self.name]:
            meta_path_emb[mp] = torch.from_numpy(self.mp_emb_dict[mp]).type(torch.FloatTensor)
        return features, adj, meta_path_emb, train_idx, train_y, val_idx, val_y, test_idx, test_y

    def get_label(self):
        train_idx = torch.from_numpy(np.array(self.labels[0])[:, 0]).type(torch.LongTensor)
        train_y = torch.from_numpy(np.array(self.labels[0])[:, 1]).type(torch.LongTensor)
        val_idx = torch.from_numpy(np.array(self.labels[1])[:, 0]).type(torch.LongTensor)
        val_y = torch.from_numpy(np.array(self.labels[1])[:, 1]).type(torch.LongTensor)
        test_idx = torch.from_numpy(np.array(self.labels[2])[:, 0]).type(torch.LongTensor)
        test_y = torch.from_numpy(np.array(self.labels[2])[:, 1]).type(torch.LongTensor)
        return train_idx, train_y, val_idx, val_y, test_idx, test_y

    def load_meta_path_emb(self):
        '''Load pretrained mp_embedding'''
        for mp in self.mp_list[self.name]:
            f_name = f'{self.data_folder}{mp}_emb.pkl'
            with open(f_name, 'rb') as f:
                z = pickle.load(f)
                zero_lines = np.nonzero(np.sum(z, 1) == 0)
                if len(zero_lines) > 0:
                    z[zero_lines, :] += 1e-8
                self.mp_emb_dict[mp] = z

    def onehot(self, labels):
        eye = np.identity(labels.max() + 1)
        onehot_mx = eye[labels.cpu()]
        return onehot_mx

    def to(self, device):
        if hasattr(self, 'g'):
            self.g = self.g.to(device)
        self.adj = self.adj.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.train_idx = self.train_idx.to(device)
        self.val_idx = self.val_idx.to(device)
        self.test_idx = self.test_idx.to(device)
        self.train_y = self.train_y.to(device)
        self.val_y = self.val_y.to(device)
        self.test_y = self.test_y.to(device)
        for k in self.meta_path_emb.keys():
            self.meta_path_emb[k] = self.meta_path_emb[k].to(device)
        return self

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2})'.format(self.name, self.adj.shape, self.features.shape)


if __name__ == '__main__':
    # from GSL.data import HeteroDataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = osp.join(osp.expanduser('~'), 'datasets')
    data = HeteroDataset(root=data_path, name='dblp', device=device)
    adj, features, labels = data.adj, data.features, data.labels
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    meta_path_emb = data.meta_path_emb
