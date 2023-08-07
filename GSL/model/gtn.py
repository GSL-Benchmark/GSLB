import time
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
from dgl.nn import HeteroEmbedding, HeteroLinear
from GSL.model import BaseModel
from GSL.utils import *


class GTN(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, data):
        super(GTN, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device)
        self.num_channels = self.config.num_channels
        self.in_dim = self.config.hid_dim
        self.hid_dim = self.config.hid_dim
        self.num_class = data.num_class
        self.num_layers = self.config.num_layers
        self.is_norm = self.config.is_norm
        self.category = data.target_ntype
        self.identity = self.config.identity
        num_edge_type = len(data.edges)
        if self.identity:
            num_edge_type += 1
        self.num_edge_type = num_edge_type
        self.hg = data.g

        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge_type, self.num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge_type, self.num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn = GraphConv(in_feats=self.in_dim, out_feats=self.hid_dim, norm='none', activation=F.relu)
        self.norm = EdgeWeightNorm(norm='right')
        self.linear1 = nn.Linear(self.hid_dim * self.num_channels, self.hid_dim)
        self.linear2 = nn.Linear(self.hid_dim, self.num_class)
        self.category_idx = None
        self.A = None
        self.h = None

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            g = dgl.remove_self_loop(g)
            g.edata['w_sum'] = self.norm(g, g.edata['w_sum'])
            norm_H.append(g)
        return norm_H

    def forward(self, hg, h):
        with hg.local_scope():
            hg.ndata['h'] = h
            # * =============== Extract edges in original graph ================
            if self.category_idx is None:
                self.A, h, self.category_idx = transform_relation_graph_list(hg, category=self.category,
                                                                             identity=self.identity)
            else:
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
            
            A = self.A
            # * =============== Get new graph structure ================
            for i in range(self.num_layers):
                if i == 0:
                    H, W = self.layers[i](A)
                else:
                    H, W = self.layers[i](A, H)
                if self.is_norm == True:
                    H = self.normalization(H)
                # Ws.append(W)
            # * =============== GCN Encoder ================
            for i in range(self.num_channels):
                g = dgl.remove_self_loop(H[i])
                edge_weight = g.edata['w_sum']
                g = dgl.add_self_loop(g)
                edge_weight = torch.cat((edge_weight, torch.full((g.number_of_nodes(),), 1, device=g.device)))
                edge_weight = self.norm(g, edge_weight)
                if i == 0:
                    X_ = self.gcn(g, h, edge_weight=edge_weight)
                else:
                    X_ = torch.cat((X_, self.gcn(g, h, edge_weight=edge_weight)), dim=1)
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return {self.category: y[self.category_idx]}

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
        self.optimizer = torch.optim.Adam([{'params': self.gcn.parameters()},
                                           {'params': self.linear1.parameters()},
                                           {'params': self.linear2.parameters()},
                                           {"params": self.layers.parameters(), "lr": 0.5}
                                           ], lr=self.config.lr, weight_decay=float(self.config.weight_decay))
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

class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first:
            result_A = self.conv1(A)
            result_B = self.conv2(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        H = []
        for i in range(len(result_A)):
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W


class GTConv(nn.Module):
    def __init__(self, in_channels, out_channels, softmax_flag=True):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.softmax_flag = softmax_flag
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, A):
        if self.softmax_flag:
            Filter = F.softmax(self.weight, dim=1)
        else:
            Filter = self.weight
        num_channels = Filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, g in enumerate(A):
                A[j].edata['w_sum'] = g.edata['w'] * Filter[i][j]
            sum_g = dgl.adj_sum_graph(A, 'w_sum')
            results.append(sum_g)
        return results

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