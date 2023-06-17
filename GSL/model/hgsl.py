import time
from GSL.model import BaseModel
from GSL.utils import *
from GSL.learner import *
from GSL.encoder import *
from GSL.metric import *
from GSL.processor import *
import torch
import torch.nn as nn


class HGSL(BaseModel):
    def __init__(self, device, config, metadata):
        super(HGSL, self).__init__(device)
        self.config = config
        self.ti, self.ri, self.types, self.ud_rels = metadata['t_info'], metadata['r_info']\
            , metadata['types'], metadata['undirected_relations']
        feat_dim, mp_emb_dim, n_class = metadata['n_feat'], metadata['n_meta_path_emb'], metadata['n_class']
        self.non_linear = nn.ReLU()
        # ! Graph Structure Learning
        MD = nn.ModuleDict
        self.fgg_direct, self.fgg_left, self.fgg_right, self.fg_agg, self.sgg_gen, self.sg_agg, self.overall_g_agg = \
            MD({}), MD({}), MD({}), MD({}), MD({}), MD({}), MD({})
        # Feature encoder
        self.encoder = MD(dict(zip(metadata['types'], [nn.Linear(feat_dim, config.com_feat_dim)
                                                       for _ in metadata['types']])))

        for r in metadata['undirected_relations']:
            # ! Feature Graph Generator
            self.fgg_direct[r] = KHeadHPLearner(CosineSimilarity(), [ThresholdSparsify(config.fgd_th)], config.com_feat_dim, config.num_head)
            self.fgg_left[r] = KHeadHPLearner(CosineSimilarity(), [ThresholdSparsify(config.fgh_th)], feat_dim, config.num_head)
            self.fgg_right[r] = KHeadHPLearner(CosineSimilarity(), [ThresholdSparsify(config.fgh_th)], feat_dim, config.num_head)
            self.fg_agg[r] = GraphChannelAttLayer(3)  # 3 = 1 (first-order/direct) + 2 (second-order)

            # ! Semantic Graph Generator
            self.sgg_gen[r] = MD(dict(
                zip(config.mp_list, [KHeadHPLearner(CosineSimilarity(), [ThresholdSparsify(config.sem_th)], mp_emb_dim, config.num_head) for _ in config.mp_list])))
            self.sg_agg[r] = GraphChannelAttLayer(len(config.mp_list))

            # ! Overall Graph Generator
            self.overall_g_agg[r] = GraphChannelAttLayer(3, [1, 1, 10])  # 3 = feat-graph + sem-graph + ori_graph

        # ! Graph Convolution
        if config.conv_method == 'gcn':
            self.GCN = GCN(in_channels=feat_dim, hidden_channels=config.emb_dim, out_channels=n_class, num_layers=2,
                           dropout=config.dropout, dropout_adj=0., sparse=False, activation_last='log_softmax',
                           conv_bias=True)
        self.norm_order = config.adj_norm_order
        self.stopper = None

    def fit(self, adj, features, labels, train_idx, val_idx, test_idx, meta_path_emb):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, weight_decay=float(self.config.weight_decay))
        self.stopper = EarlyStopping(patience=self.config.early_stop)
        cla_loss = torch.nn.NLLLoss()

        dur = []
        # w_list = []
        for epoch in range(self.config.epochs):
            # ! Train
            t0 = time.time()
            self.train()
            logits, adj_new = self.forward(features, adj, meta_path_emb)
            train_f1, train_mif1 = self.eval_logits(logits, train_idx, labels[train_idx])

            l_pred = cla_loss(logits[train_idx], labels[train_idx])
            l_reg = self.config.alpha * torch.norm(adj, 1)
            loss = l_pred + l_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ! Valid
            self.eval()
            with torch.no_grad():
                logits = self.GCN(features, adj_new)
                val_f1, val_mif1 = self.eval_logits(logits, val_idx, labels[val_idx])
            dur.append(time.time() - t0)
            print(
                f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Loss {loss.item():.4f} | TrainF1 {train_f1:.4f} | ValF1 {val_f1:.4f}")

            if self.config.early_stop > 0:
                if self.stopper.step(val_f1, self, epoch):
                    print(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break

        if self.config.early_stop > 0:
            self.load_state_dict(self.stopper.best_weight)

    def test(self, adj, features, labels, train_idx, val_idx, test_idx, meta_path_emb):
        with torch.no_grad():
            logits, _ = self.forward(features, adj, meta_path_emb)
            self.eval_and_save(logits, test_idx, labels[test_idx], val_idx, labels[val_idx])

    def eval_logits(self, logits, target_x, target_y):
        pred_y = torch.argmax(logits[target_x], dim=1)
        return torch_f1_score(pred_y, target_y, n_class=logits.shape[1])

    def eval_and_save(self, logits, test_x, test_y, val_x, val_y, res={}):
        test_f1, test_mif1 = self.eval_logits(logits, test_x, test_y)
        val_f1, val_mif1 = self.eval_logits(logits, val_x, val_y)
        self.save_results(test_f1, val_f1, test_mif1, val_mif1, res)

    def save_results(self, test_f1, val_f1, test_mif1=0, val_mif1=0, res={}):
        if self.stopper != None:
            res.update({'test_f1': f'{test_f1:.4f}', 'test_mif1': f'{test_mif1:.4f}',
                        'val_f1': f'{val_f1:.4f}', 'val_mif1': f'{val_mif1:.4f}',
                        'best_epoch': self.stopper.best_epoch})
        else:
            res.update({'test_f1': f'{test_f1:.4f}', 'test_mif1': f'{test_mif1:.4f}',
                        'val_f1': f'{val_f1:.4f}', 'val_mif1': f'{val_mif1:.4f}'})
        # print(f"Seed{self.config.seed}")
        res_dict = {'res': res}
        print(f'\n\n\nTrain finished, results:{res_dict}')

    def forward(self, features, adj_ori, mp_emb):
        def get_rel_mat(mat, r):
            return mat[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]]

        def get_type_rows(mat, type):
            return mat[self.ti[type]['ind'], :]

        def gen_g_via_feat(graph_gen_func, mat, r):
            return graph_gen_func(get_type_rows(mat, r[0]), get_type_rows(mat, r[-1]))

        # ! Heterogeneous Feature Mapping
        com_feat_mat = torch.cat([self.non_linear(
            self.encoder[t](features[self.ti[t]['ind']])) for t in self.types])

        # ! Heterogeneous Graph Generation
        new_adj = torch.zeros_like(adj_ori).to(self.device)
        for r in self.ud_rels:
            ori_g = get_rel_mat(adj_ori, r)
            # ! Feature Graph Generation
            fg_direct = gen_g_via_feat(self.fgg_direct[r], com_feat_mat, r)

            fmat_l, fmat_r = features[self.ti[r[0]]['ind']], features[self.ti[r[-1]]['ind']]
            sim_l, sim_r = self.fgg_left[r](fmat_l, fmat_l), self.fgg_right[r](fmat_r, fmat_r)
            fg_left, fg_right = sim_l.mm(ori_g), sim_r.mm(ori_g.t()).t()

            feat_g = self.fg_agg[r]([fg_direct, fg_left, fg_right])

            # ! Semantic Graph Generation
            sem_g_list = [gen_g_via_feat(self.sgg_gen[r][mp], mp_emb[mp], r) for mp in mp_emb]
            sem_g = self.sg_agg[r](sem_g_list)
            # ! Overall Graph
            # Update relation sub-matixs
            new_adj[self.ri[r][0]:self.ri[r][1], self.ri[r][2]:self.ri[r][3]] = \
                self.overall_g_agg[r]([feat_g, sem_g, ori_g])  # update edge  e.g. AP

        new_adj += new_adj.clone().t()  # sysmetric
        # ! Aggregate
        new_adj = F.normalize(new_adj, dim=0, p=self.norm_order)
        logits = self.GCN(features, new_adj)
        return logits, new_adj


class GraphChannelAttLayer(nn.Module):
    """
    Fuse a multi-channel graph to a single-channel graph with attention.
    """
    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        adj_list = F.normalize(adj_list, dim=1, p=1)
        # Hadamard product + summation -> Conv
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)
