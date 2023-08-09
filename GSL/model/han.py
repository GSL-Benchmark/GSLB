import dgl
import torch.nn as nn
from dgl.nn import HeteroEmbedding, HeteroLinear
from dgl.nn.pytorch import GATConv
from GSL.model import BaseModel
from GSL.utils import *


class HAN(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, data):
        super(HAN, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device)
        self.update_config()

        self.in_dim = self.config.hid_dim
        self.hid_dim = self.config.hid_dim
        self.num_heads = self.config.num_heads
        self.dropout = self.config.dropout

        self.config.meta_paths_dict = data.meta_paths_dict

        self.category = data.target_ntype
        self.num_class = data.num_class
        self.out_dim = self.num_class
        self.config.category = self.category
        self.config.out_node_type = [self.category]

        self.hg = data.g

        ntypes = set()
        ntypes.add(self.config.category)

        ntype_meta_paths_dict = {}
        for ntype in ntypes:
            ntype_meta_paths_dict[ntype] = {}
            for meta_path_name, meta_path in self.config.meta_paths_dict.items():
                # a meta path starts with this node type
                if meta_path[0][0] == ntype:
                    ntype_meta_paths_dict[ntype][meta_path_name] = meta_path
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                ntype_meta_paths_dict[ntype] = self.extract_metapaths(ntype, self.hg.canonical_etypes)

        self.mod_dict = nn.ModuleDict()
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            self.mod_dict[ntype] = _HAN(meta_paths_dict, self.in_dim, self.hid_dim, self.out_dim, self.num_heads, self.dropout)

    def update_config(self):
        # if self.config.dataset == 'acm':
        self.config.num_heads = [1]

    def extract_metapaths(self, category, canonical_etypes, self_loop=False):
        meta_paths_dict = {}
        for etype in canonical_etypes:
            if etype[0] in category:
                for dst_e in canonical_etypes:
                    if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                        if self_loop:
                            mp_name = 'mp' + str(len(meta_paths_dict))
                            meta_paths_dict[mp_name] = [etype, dst_e]
                        else:
                            if etype[0] != etype[2]:
                                mp_name = 'mp' + str(len(meta_paths_dict))
                                meta_paths_dict[mp_name] = [etype, dst_e]
        return meta_paths_dict

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            g = dgl.remove_self_loop(g)
            g.edata['w_sum'] = self.norm(g, g.edata['w_sum'])
            norm_H.append(g)
        return norm_H

    def forward(self, g, h_dict):
        r"""
        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, dict[str, DGLBlock]]
            For full batch, it is a heterogeneous graph. For mini batch, it is a dict from node type to dict from
            mata path name to DGLBlock.
        h_dict : dict[str, Tensor] or dict[str, dict[str, dict[str, Tensor]]]
            The input features. For full batch, it is a dict from node type to node features. For mini batch, it is
            a dict from node type to dict from meta path name to dict from node type to node features.

        Returns
        --------
        out_dict : dict[str, Tensor]
            The output features. Dict from node type to node features.
        """
        out_dict = {}
        for ntype, han in self.mod_dict.items():
            if isinstance(g, dict):
                # mini batch
                if ntype not in g:
                    continue
                _g = g[ntype]
                _in_h = h_dict[ntype]
            else:
                # full batch
                _g = g
                _in_h = h_dict
            _out_h = han(_g, _in_h)
            for ntype, h in _out_h.items():
                out_dict[ntype] = h

        return out_dict

    def eval_logits(self, logits, target_x, target_y):
        pred_y = torch.argmax(logits[target_x], dim=1)
        return macro_f1(pred_y, target_y, n_class=logits.shape[1]), micro_f1(pred_y, target_y, n_class=logits.shape[1])

    def init_feature(self, act):
        # self.logger.feature_info("Feat is 0, nothing to do!")
        if isinstance(self.hg.ndata['h'], dict):
            # The heterogeneous contains more than one node type.
            input_feature = HeteroFeature(self.hg.ndata['h'], get_nodes_dict(self.hg),
                                            self.config.hid_dim, act=act).to(self.device)
        elif isinstance(self.hg.ndata['h'], torch.Tensor):
            # The heterogeneous only contains one node type.
            input_feature = HeteroFeature({self.hg.ntypes[0]: self.hg.ndata['h']}, get_nodes_dict(self.hg),
                                            self.config.hid_dim, act=act).to(self.device)
        return input_feature

    def fit(self, data):
        labels, train_idx, val_idx, test_idx = data.labels, data.train_idx, data.val_idx, data.test_idx
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=float(self.config.weight_decay))

        self.input_feature = self.init_feature(None)
        self.optimizer.add_param_group({'params': self.input_feature.parameters()})
        self.add_module('input_feature', self.input_feature)

        self.stopper = EarlyStopping(patience=self.config.early_stop)

        cla_loss = F.cross_entropy

        dur = []
        for epoch in range(self.config.epochs):
            # ! Train
            t0 = time.time()
            self.train()
            h_dict = self.input_feature()
            logits = self.forward(self.hg, h_dict)[self.category]
            train_f1, train_mif1 = self.eval_logits(logits, train_idx, labels[train_idx])

            loss = cla_loss(logits[train_idx], labels[train_idx])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ! Valid
            self.eval()
            with torch.no_grad():
                h_dict = self.input_feature()
                h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
                logits = self.forward(self.hg, h_dict)[self.category]
                val_f1, val_mif1 = self.eval_logits(logits, val_idx, labels[val_idx])
                val_loss = cla_loss(logits[val_idx], labels[val_idx]).item()
            dur.append(time.time() - t0)
            print(
                f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Train Loss {loss.item():.4f} | TrainF1 {train_f1:.4f} | ValF1 {val_f1:.4f}")

            if self.config.early_stop > 0:
                if self.stopper.loss_step(val_loss, self, epoch):
                    print(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break

        if self.config.early_stop > 0:
            self.load_state_dict(self.stopper.best_weight)
        self.test(labels, val_idx, test_idx)

    def test(self, labels, val_idx, test_idx):
        with torch.no_grad():
            h_dict = self.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = self.forward(self.hg, h_dict)[self.category]
            test_f1, test_mif1 = self.eval_logits(logits, test_idx, labels[test_idx])
            val_f1, val_mif1 = self.eval_logits(logits, val_idx, labels[val_idx])
            res = {}
            if self.stopper != None:
                res.update({'test_f1': f'{test_f1:.4f}', 'test_mif1': f'{test_mif1:.4f}',
                            'val_f1': f'{val_f1:.4f}', 'val_mif1': f'{val_mif1:.4f}',
                            'best_epoch': self.stopper.best_epoch})
            else:
                res.update({'test_f1': f'{test_f1:.4f}', 'test_mif1': f'{test_mif1:.4f}',
                            'val_f1': f'{val_f1:.4f}', 'val_mif1': f'{val_mif1:.4f}'})
            # print(f"Seed{self.config.seed}")
            res_dict = {'res': res}
            print(f'results:{res_dict}')
            self.best_result = test_f1.item()


class _HAN(nn.Module):

    def __init__(self, meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(_HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths_dict, in_dim, hidden_dim, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths_dict, hidden_dim * num_heads[l - 1],
                                        hidden_dim, num_heads[l], dropout))
        self.linear = nn.Linear(hidden_dim * num_heads[-1], out_dim)

    def forward(self, g, h_dict):
        for gnn in self.layers:
            h_dict = gnn(g, h_dict)
        out_dict = {}
        for ntype, h in h_dict.items():  # only one ntype here
            out_dict[ntype] = self.linear(h_dict[ntype])
        return out_dict

    def get_emb(self, g, h_dict):
        h = h_dict[self.category]
        for gnn in self.layers:
            h = gnn(g, h)

        return {self.category: h.detach().cpu().numpy()}


class HANLayer(nn.Module):
    """
    HAN layer.

    Parameters
    ------------
    meta_paths_dict : dict[str, list[etype]]
        Dict from meta path name to meta path.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output feature dimension.
    layer_num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.

    Attributes
    ------------
    _cached_graph : dgl.DGLHeteroGraph
        a cached graph
    _cached_coalesced_graph : list
        _cached_coalesced_graph list generated by *dgl.metapath_reachable_graph()*
    """

    def __init__(self, meta_paths_dict, in_dim, out_dim, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.meta_paths_dict = meta_paths_dict
        semantic_attention = SemanticAttention(in_size=out_dim * layer_num_heads)
        mods = nn.ModuleDict({mp: GATConv(in_dim, out_dim, layer_num_heads,
                                          dropout, dropout, activation=F.elu,
                                          allow_zero_in_degree=True) for mp in meta_paths_dict})
        self.model = MetapathConv(meta_paths_dict, mods, semantic_attention)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        r"""
        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, DGLBlock]
            For full batch, it is a heterogeneous graph. For mini batch, it is a dict from mata path name to DGLBlock.
        h : dict[str, Tensor] or dict[str, dict[str, Tensor]]
            The input features. For full batch, it is a dict from node type to node features. For mini batch, it is
            a  dict from meta path name to dict from node type to node features.

        Returns
        --------
        h : dict[str, Tensor]
            The output features. Dict from node type to node features. Only one node type.
        """
        # mini batch
        if isinstance(g, dict):
            h = self.model(g, h)

        # full batch
        else:
            if self._cached_graph is None or self._cached_graph is not g:
                self._cached_graph = g
                self._cached_coalesced_graph.clear()
                for mp, mp_value in self.meta_paths_dict.items():
                    self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(
                        g, mp_value)
            h = self.model(self._cached_coalesced_graph, h)

        return h


class HeteroFeature(nn.Module):
    r"""
    This is a feature preprocessing component which is dealt with various heterogeneous feature situation.

    In general, we will face the following three situations.

        1. The dataset has not feature at all.

        2. The dataset has features in every node type.

        3. The dataset has features of a part of node types.

    To deal with that, we implement the HeteroFeature.In every situation, we can see that

        1. We will build embeddings for all node types.

        2. We will build linear layer for all node types.

        3. We will build embeddings for parts of node types and linear layer for parts of node types which have original feature.

    Parameters
    ----------
    h_dict: dict
        Input heterogeneous feature dict,
        key of dict means node type,
        value of dict means corresponding feature of the node type.
        It can be None if the dataset has no feature.
    n_nodes_dict: dict
        Key of dict means node type,
        value of dict means number of nodes.
    embed_size: int
        Dimension of embedding, and used to assign to the output dimension of Linear which transform the original feature.
    need_trans: bool, optional
        A flag to control whether to transform original feature linearly. Default is ``True``.
    act : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    -----------
    embed_dict : nn.ParameterDict
        store the embeddings

    hetero_linear : HeteroLinearLayer
        A heterogeneous linear layer to transform original feature.
    """

    def __init__(self, h_dict, n_nodes_dict, embed_size, act=None, need_trans=True, all_feats=True):
        super(HeteroFeature, self).__init__()
        self.n_nodes_dict = n_nodes_dict
        self.embed_size = embed_size
        self.h_dict = h_dict
        self.need_trans = need_trans

        self.type_node_num_sum = [0]
        self.all_type = []
        for ntype, type_num in n_nodes_dict.items():
            num_now = self.type_node_num_sum[-1]
            num_now += type_num
            self.type_node_num_sum.append(num_now)
            self.all_type.append(ntype)
        self.type_node_num_sum = torch.tensor(self.type_node_num_sum)

        linear_dict = {}
        embed_dict = {}
        for ntype, n_nodes in self.n_nodes_dict.items():
            h = h_dict.get(ntype)
            if h is None:
                if all_feats:
                    embed_dict[ntype] = n_nodes
            else:
                linear_dict[ntype] = h.shape[1]
        self.embes = HeteroEmbedding(embed_dict, embed_size)
        if need_trans:
            self.linear = HeteroLinear(linear_dict, embed_size)
        self.act = act  # activate

    def forward(self):
        out_dict = {}
        out_dict.update(self.embes.weight)
        tmp = self.linear(self.h_dict)
        if self.act:  # activate
            for x, y in tmp.items():
                tmp.update({x: self.act(y)})
        out_dict.update(tmp)
        return out_dict

    def forward_nodes(self, id_dict):
        # Turn "id_dict" into a dictionary if "id_dict" is a tensor, and record the corresponding relationship in "to_pos"
        id_tensor = None
        if torch.is_tensor(id_dict):
            device = id_dict.device
        else:
            device = id_dict.get(next(iter(id_dict))).device

        if torch.is_tensor(id_dict):
            id_tensor = id_dict
            self.type_node_num_sum = self.type_node_num_sum.to(device)
            id_dict = {}
            to_pos = {}
            for i, x in enumerate(id_tensor):
                tmp = torch.where(self.type_node_num_sum <= x)[0]
                if len(tmp) > 0:
                    tmp = tmp.max()
                    now_type = self.all_type[tmp]
                    now_id = x - self.type_node_num_sum[tmp]
                    if now_type not in id_dict.keys():
                        id_dict[now_type] = []
                    id_dict[now_type].append(now_id)
                    if now_type not in to_pos.keys():
                        to_pos[now_type] = []
                    to_pos[now_type].append(i)
            for ntype in id_dict.keys():
                id_dict[ntype] = torch.tensor(id_dict[ntype], device=device)

        embed_id_dict = {}
        linear_id_dict = {}
        for entype, id in id_dict.items():
            if self.h_dict.get(entype) is None:
                embed_id_dict[entype] = id
            else:
                linear_id_dict[entype] = id
        out_dict = {}
        tmp = self.embes(embed_id_dict)
        out_dict.update(tmp)
        # for key in self.h_dict:
        #     self.h_dict[key] = self.h_dict[key].to(device)
        h_dict = {}
        for key in linear_id_dict:
            linear_id_dict[key] = linear_id_dict[key].to('cpu')
        for key in linear_id_dict:
            h_dict[key] = self.h_dict[key][linear_id_dict[key]].to(device)
        tmp = self.linear(h_dict)
        if self.act:  # activate
            for x, y in tmp.items():
                tmp.update({x: self.act(y)})
        for entype in linear_id_dict:
            out_dict[entype] = tmp[entype]

        # The result corresponds to the original position according to the corresponding relationship
        if id_tensor is not None:
            out_feat = [None] * len(id_tensor)
            for ntype, feat_list in out_dict.items():
                for i, feat in enumerate(feat_list):
                    now_pos = to_pos[ntype][i]
                    out_feat[now_pos] = feat.data
            out_dict = torch.stack(out_feat, dim=0)

        return out_dict

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, nty=None):
        if len(z) == 0:
            return None
        z = torch.stack(z, dim=1)
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class MetapathConv(nn.Module):
    r"""
    MetapathConv is an aggregation function based on meta-path, which is similar with `dgl.nn.pytorch.HeteroGraphConv`.
    We could choose Attention/ APPNP or any GraphConvLayer to aggregate node features.
    After that we will get embeddings based on different meta-paths and fusion them.

    .. math::
        \mathbf{Z}=\mathcal{F}(Z^{\Phi_1},Z^{\Phi_2},...,Z^{\Phi_p})=\mathcal{F}(f(H,\Phi_1),f(H,\Phi_2),...,f(H,\Phi_p))

    where :math:`\mathcal{F}` denotes semantic fusion function, such as semantic-attention. :math:`\Phi_i` denotes meta-path and
    :math:`f` denotes the aggregation function, such as GAT, APPNP.

    Parameters
    ------------
    meta_paths_dict : dict[str, list[tuple(meta-path)]]
        contain multiple meta-paths.
    mods : nn.ModuleDict
        aggregation function
    macro_func : callable aggregation func
        A semantic aggregation way, e.g. 'mean', 'max', 'sum' or 'attention'

    """

    def __init__(self, meta_paths_dict, mods, macro_func, **kargs):
        super(MetapathConv, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.mods = mods
        self.meta_paths_dict = meta_paths_dict
        self.SemanticConv = macro_func

    def forward(self, g_dict, h_dict):
        r"""
        Parameters
        -----------
        g_dict : dict[str: dgl.DGLGraph]
            A dict of DGLGraph(full batch) or DGLBlock(mini batch) extracted by metapaths.
        h_dict : dict[str: torch.Tensor]
            The input features

        Returns
        --------
        h : dict[str: torch.Tensor]
            The output features dict
        """
        outputs = {g.dsttypes[0]: [] for s, g in g_dict.items()}

        for meta_path_name, meta_path in self.meta_paths_dict.items():
            new_g = g_dict[meta_path_name]

            # han minibatch
            if h_dict.get(meta_path_name) is not None:
                h = h_dict[meta_path_name][new_g.srctypes[0]]
            # full batch
            else:
                h = h_dict[new_g.srctypes[0]]
            outputs[new_g.dsttypes[0]].append(self.mods[meta_path_name](new_g, h).flatten(1))
        # semantic_embeddings = th.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        # Aggregate the results for each destination node type
        rsts = {}
        for ntype, ntype_outputs in outputs.items():
            if len(ntype_outputs) != 0:
                rsts[ntype] = self.SemanticConv(ntype_outputs)  # (N, D * K)
        return rsts
