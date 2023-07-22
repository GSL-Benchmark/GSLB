import os
import random
import time
from abc import ABCMeta, abstractmethod

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from gensim.models import Word2Vec
from scipy.sparse import coo_matrix
from tqdm import tqdm, trange

from GSL.encoder.gcn import GCN, GCNConv, GCNConv_diag
from GSL.encoder.mlp import MLP
from GSL.metric import InnerProductSimilarity
from GSL.model import BaseModel
from GSL.utils import (EarlyStopping, Evaluator, MemoryMoCo, NCESoftmaxLoss,
                       PolynomialLRDecay, accuracy, edge_lists_to_set,
                       get_cur_time, global_topk, graph_edge_to_lot,
                       load_pickle, min_max_scaling, mkdir_list, moment_update,
                       para_copy, save_pickle, time2str, time_logger)

P_EPOCHS_SAVE_LIST = [1, 2, 3, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300]

EOS = 1e-10


def gen_dw_emb(A, number_walks=10, alpha=0, walk_length=100, window=10, workers=16, size=128):  # ,row, col,
    row, col = A.nonzero()
    edges = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1).astype(dtype=np.dtype(str))
    print("build adj_mat")
    t1 = time.time()
    G = {}
    for [i, j] in edges:
        if i not in G:
            G[i] = []
        if j not in G:
            G[j] = []
        G[i].append(j)
        G[j].append(i)
    for node in G:
        G[node] = list(sorted(set(G[node])))
        if node in G[node]:
            G[node].remove(node)

    nodes = list(sorted(G.keys()))
    print("len(G.keys()):", len(G.keys()), "\tnode_num:", A.shape[0])
    corpus = []  # 存放上下文的 list,每一个节点一个上下文(随机游走序列)
    for cnt in trange(number_walks):
        random.shuffle(nodes)
        for idx, node in enumerate(nodes):
            path = [node]  # 对每个节点找到他的游走序列.
            while len(path) < walk_length:
                cur = path[-1]  # 每次从序列的尾记录当前游走位置.
                if len(G[cur]) > 0:
                    if random.random() >= alpha:
                        path.append(random.choice(G[cur]))  # 如果有邻居,邻接矩阵里随便选一个
                    else:
                        path.append(path[0])  # Random Walk with restart
                else:
                    break
            corpus.append(path)
    t2 = time.time()
    print(f"Corpus generated, time cost: {time2str(t2 - t1)}")
    print("Training word2vec")
    model = Word2Vec(corpus,
                     vector_size=size,  # emb_size
                     window=window,
                     min_count=0,
                     sg=1,  # skip gram
                     hs=1,  # hierarchical softmax
                     workers=workers)
    print("done.., cost: {}s".format(time.time() - t2))
    output = []
    for i in range(A.shape[0]):
        if str(i) in model.wv:  # word2vec 的输出以字典的形式存在.wv 里
            output.append(model.wv[str(i)])
        else:
            print("{} not trained".format(i))
            output.append(np.zeros(size))
    return np.array(output)


def train_deepwalk(cf, g):
    # g = coo_matrix(g.cpu().numpy())
    # g = dgl.from_scipy(g)
    
    # g = g.to(cf.device)
    adj = g.adj_external(scipy_fmt='coo')
    # adj = g # adj need to be scipy coo sparse matrix
    
    emb_mat = gen_dw_emb(adj, number_walks=cf.se_num_walks, walk_length=cf.se_walk_length, window=cf.se_window_size,
                         size=cf.se_n_hidden,
                         workers=cf.se_num_workers)

    emb_mat = torch.FloatTensor(emb_mat)
    return emb_mat


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-IDF
    import random

    import dgl
    import torch
    dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def cosine_sim_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    
    
def cosine_similarity_n_space(m1=None, m2=None, dist_batch_size=100):
    NoneType = type(None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if type(m1) != torch.Tensor:  # only numpy conversion supported
        m1 = torch.from_numpy(m1).float()
    if type(m2) != torch.Tensor and type(m2) != NoneType:
        m2 = torch.from_numpy(m2).float()  # m2 could be None

    m2 = m1 if m2 is None else m2
    assert m1.shape[1] == m2.shape[1]

    result = torch.zeros([1, m2.shape[0]])

    for row_i in tqdm(range(0, int(m1.shape[0] / dist_batch_size) + 1), desc='Calculating pairwise similarity'):
        start = row_i * dist_batch_size
        end = min([(row_i + 1) * dist_batch_size, m1.shape[0]])
        if end <= start:
            break
        rows = m1[start: end]
        # sim = cosine_similarity(rows, m2) # rows is O(1) size
        sim = cosine_sim_torch(rows.to(device), m2.to(device))

        result = torch.cat((result, sim.cpu()), 0)

    result = result[1:, :]  # deleting the first row, as it was used for setting the size only
    del sim
    return result  # return 1 - ret # should be used with sklearn cosine_similarity


def print_log(log_dict):
    log_ = lambda log: f'{log:.4f}' if isinstance(log, float) else f'{log:04d}'
    print(' | '.join([f'{k} {log_(v)}' for k, v in log_dict.items()]))


class GSR_pretrain(nn.Module):
    def __init__(self, g, config):
        # ! Initialize variabless
        super(GSR_pretrain, self).__init__()
        self.__dict__.update(config.model_conf)
        init_random_state(config.seed)
        self.device = config.device
        self.g = g

        # ! Encoder: Pretrained GNN Modules
        self.views = views = ['F', 'S']
        self.encoder = nn.ModuleDict({
            src_view: self._get_encoder(src_view, config) for src_view in views})
        # ! Decoder: Cross-view MLP Mappers
        self.decoder = nn.ModuleDict(
            {f'{src_view}->{tgt_view}':
                 MLP(n_layer=self.decoder_layer,
                     input_dim=self.n_hidden,
                     n_hidden=self.decoder_n_hidden,
                     output_dim=self.n_hidden, dropout=0,
                     activation=nn.ELU(),
                     )
             for tgt_view in views
             for src_view in views if src_view != tgt_view})

    def _get_encoder(self, src_view, config):
        input_dim = config.feat_dim[src_view]
        if config.gnn_model == 'GCN':
            # GCN emb should not be dropped out
            return GCN(input_dim, config.n_hidden, config.n_hidden, 2, config.pre_dropout, 0, False, activation_last=False)
            # return TwoLayerGCN(input_dim, config.n_hidden, config.n_hidden, config.activation, config.pre_dropout, is_out_layer=True)
        # if config.gnn_model == 'GAT':
        #     return TwoLayerGAT(input_dim, config.gat_hidden, config.gat_hidden, config.in_head, config.prt_out_head, config.activation, config.pre_dropout, config.pre_dropout,  is_out_layer=True)
        # if config.gnn_model == 'GraphSage':
        #     return TwoLayerGraphSage(input_dim, config.n_hidden, config.n_hidden, config.aggregator, config.activation, config.pre_dropout, is_out_layer=True)
        # if config.gnn_model == 'SGC':
        #     return OneLayerSGC(input_dim, config.n_hidden, k=config.k, is_out_layer=True)
        # if config.gnn_model == 'GCNII':
        #     return TwoLayerGCNII(input_dim, config.n_hidden, config.n_hidden, config.activation, config.pre_dropout, config.alpha, config.lda, is_out_layer=True)

    def forward(self, edge_subgraph, blocks, input, mode='q'):
        def _get_emb(x):
            # Query embedding is stored in source nodes, key embedding in target
            q_nodes, k_nodes = edge_subgraph.edges()
            return x[q_nodes] if mode == 'q' else x[k_nodes]

        # ! Step 1: Encode node properties to embeddings
        Z = {src_view: _get_emb(encoder(blocks, input[src_view], stochastic=True))
             for src_view, encoder in self.encoder.items()}
        # ! Step 2: Decode embeddings if inter-view
        Z.update({dec: decoder(Z[dec[0]])
                  for dec, decoder in self.decoder.items()})
        return Z

    @time_logger
    def refine_graph(self, g, feat):
        '''
        Find the neighborhood candidates for each candidate graph
        :param g: DGL graph
        '''

        # # ! Get Node Property
        emb = {_: self.encoder[_](g.to(self.device), feat[_].to(self.device), stochastic=False).detach()
               for _ in self.views}
        edges = set(graph_edge_to_lot(g))
        rm_num, add_num = [int(float(_) * self.g.num_edges())
                           for _ in (self.rm_ratio, self.add_ratio)]
        batched_implementation = True
        if batched_implementation:
            # if not self.fsim_norm or self.dataset in ['arxiv']:
            if self.device != torch.device('cpu'):
                emb = {k: v.half() for k, v in emb.items()}
            edges = scalable_graph_refine(
                g, emb, rm_num, add_num, self.cos_batch_size, self.fsim_weight, self.device, self.fsim_norm)
        else:
            # ! Filter Graphs
            sim_mats = {v: cosine_similarity_n_space(np_.detach(), dist_batch_size=self.cos_batch_size)
                        for v, np_ in emb.items()}
            if self.fsim_norm:
                sim_mats = {v: min_max_scaling(sim_mat, type='global') for v, sim_mat in sim_mats.items()}
            sim_adj = self.fsim_weight * sim_mats['F'] + (1 - self.fsim_weight) * sim_mats['S']
            # ! Remove the lowest K existing edges.
            # Only the existing edges should be selected, other edges are guaranteed not to be selected with similairty 99
            if rm_num > 0:
                low_candidate_mat = torch.ones_like(sim_adj) * 99
                low_candidate_mat[self.g.edges()] = sim_adj[self.g.edges()]
                low_inds = edge_lists_to_set(global_topk(low_candidate_mat, k=rm_num, largest=False))
                edges -= low_inds
            # ! Add the highest K from non-existing edges.
            # Exisiting edges and shouldn't be selected
            if add_num > 0:
                sim_adj.masked_fill_(torch.eye(sim_adj.shape[0]).bool(), -1)
                sim_adj[self.g.edges()] = -1
                high_inds = edge_lists_to_set(global_topk(sim_adj, k=add_num, largest=True))
                edges |= high_inds
            save_pickle(sorted(edges), 'EdgesGeneratedByOriImplementation')
        row_id, col_id = map(list, zip(*list(edges)))
        # print(f'High inds {list(high_inds)[:5]}')
        g_new = dgl.add_self_loop(
            dgl.graph((row_id, col_id), num_nodes=self.g.num_nodes())).to(self.device)
        # g_new.ndata['sim'] = sim_adj.to(self.device)
        return g_new


class GSR_finetune(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self, config):
        # ! Initialize variables
        super(GSR_finetune, self).__init__()
        self.__dict__.update(config.model_conf)
        init_random_state(config.seed)
        if config.dataset != 'arxiv':
            if config.gnn_model == 'GCN':
                return GCN(config.n_fea, config.n_hidden, config.n_hidden, 2, config.pre_dropout, 0, False, activation_last=False)

                # self.gnn = TwoLayerGCN(config.n_feat, config.n_hidden, config.n_class, config.activation, config.dropout, is_out_layer=True)
            # if config.gnn_model == 'GAT':
            #     self.gnn = TwoLayerGAT(config.n_feat, config.gat_hidden, config.n_class, config.in_head, 1, config.activation, config.dropout, config.dropout, is_out_layer=True)
            # if config.gnn_model == 'GraphSage':
            #     self.gnn = TwoLayerGraphSage(config.n_feat, config.n_hidden, config.n_class, config.aggregator, config.activation, config.dropout, is_out_layer=True)
            # if config.gnn_model == 'SGC':
            #     self.gmm_model = OneLayerSGC(config.n_feat, config.n_class, k=config.k, is_out_layer=True)
            # if config.gnn_model == 'GCNII':
            #     self.gnn = TwoLayerGCNII(config.n_feat, config.n_hidden, config.n_class, config.activation, config.dropout, config.alpha, config.lda,  is_out_layer=True)
        else:
            if config.gnn_model == 'GCN':
                self.gnn = GCN(config.n_feat, config.n_hidden, config.n_class, 3, config.pre_dropout, 0, False, activation_last=False, bn=True)
            #     # self.gnn = ThreeLayerGCN_BN(config.n_feat, config.n_hidden, config.n_class, config.activation, config.dropout)
            # if config.gnn_model == 'GAT':
            #     raise NotImplementedError
            # if config.gnn_model == 'GraphSage':
            #     raise NotImplementedError
            # if config.gnn_model == 'SGC':
            #     raise NotImplementedError
            # if config.gnn_model == 'GCNII':
            #     raise NotImplementedError


    def forward(self, g, x):
        return self.gnn(g, x)


# @time_logger
def get_structural_feature(g, config):
    '''
    Get structural node property prior embedding
    '''
    print('Loading structural embedding...')

    if not os.path.exists(config.structural_em_file):
        if not os.path.exists(os.path.dirname(config.structural_em_file)):
            os.makedirs(os.path.dirname(config.structural_em_file))
        print(f'Embedding file {config.structural_em_file} not exist, start training')
        if config.semb == 'dw':
            config.load_device = torch.device('cpu')
            dw_cf = SEConfig(config)
            dw_cf.device = torch.device("cuda:0") if config.gpu >= 0 else torch.device('cpu')
            emb = train_deepwalk(dw_cf, g).to(config.device)
        else:
            raise ValueError

        torch.save(emb, config.structural_em_file)
        print('Embedding file saved')
    else:
        emb = torch.load(config.structural_em_file, map_location=torch.device('cpu'))
        # print(emb)
        # print(emb.shape)
        # print(ssss)
        print(f'Load embedding file {config.structural_em_file} successfully')
    return emb


def get_pretrain_loader(g, config):
    g = g.remove_self_loop()  # Self loops shan't be sampled
    src, dst = g.edges()
    n_edges = g.num_edges()
    train_seeds = np.arange(n_edges)
    g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))
    reverse_eids = torch.cat([torch.arange(n_edges, 2 * n_edges), torch.arange(0, n_edges)])
    # Create sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in config.fan_out.split('_')])
    return dgl.dataloading.EdgeDataLoader(
        g.cpu(), train_seeds, sampler, exclude='reverse_id',
        reverse_eids=reverse_eids,
        batch_size=config.p_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=config.num_workers)


def write_nested_dict(d, f_path):
    def _write_dict(d, f):
        for key in d.keys():
            if isinstance(d[key], dict):
                f.write(str(d[key]) + '\n')

    with open(f_path, 'a+') as f:
        f.write('\n')
        _write_dict(d, f)

def get_stochastic_loader(g, train_nids, batch_size, num_workers):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    return dgl.dataloading.NodeDataLoader(
        g.cpu(), train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

def save_results(cf, res_dict):
    print(f'\nTrain seed{cf.seed} finished\nResults:{res_dict}\nConfig: {cf}')
    res_dict = {'parameters': cf.model_conf, 'res': res_dict}
    write_nested_dict(res_dict, cf.res_file)

class NodeClassificationTrainer(metaclass=ABCMeta):
    def __init__(self, model, g, features, optimizer, stopper, loss_func, sup, cf):
        self.trainer = None
        self.model = model
        self.g = g.cpu()
        self.features = features
        self.optimizer = optimizer
        self.stopper = stopper
        self.loss_func = loss_func
        self.cf = cf
        self.device = cf.device
        self.epochs = cf.epochs
        self.n_class = cf.n_class
        self.__dict__.update(sup.__dict__)
        self.train_x, self.val_x, self.test_x = \
            [_.to(cf.device) for _ in [sup.train_x, sup.val_x, sup.test_x]]
        self.labels = sup.labels.to(cf.device)
        self._evaluator = Evaluator(name='ogbn-arxiv')
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels.view(-1, 1)}
        )["acc"]

    @abstractmethod
    def _train(self):
        return None, None

    @abstractmethod
    def _evaluate(self):
        return None, None

    def run(self):
        for epoch in range(self.epochs):
            t0 = time()
            loss, train_acc = self._train()
            val_acc, test_acc = self._evaluate()
            print_log({'Epoch': epoch, 'Time': time() - t0, 'loss': loss,
                       'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc})
            if self.stopper is not None:
                if self.stopper.step(val_acc, self.model, epoch):
                    print(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))
        return self.model

    def eval_and_save(self):
        val_acc, test_acc = self._evaluate()
        res = {'test_acc': f'{test_acc:.4f}', 'val_acc': f'{val_acc:.4f}'}
        if self.stopper is not None: res['best_epoch'] = self.stopper.best_epoch
        save_results(self.cf, res)

class FullBatchTrainer(NodeClassificationTrainer):
    def __init__(self, **kwargs):
        super(FullBatchTrainer, self).__init__(**kwargs)
        self.g = self.g.to(self.device)
        self.features = self.features.to(self.device)

    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.g, self.features)
        loss = self.loss_func(logits[self.train_x], self.labels[self.train_x])
        train_acc = self.evaluator(logits[self.train_x], self.labels[self.train_x])
        loss.backward()
        self.optimizer.step()
        return loss.item(), train_acc

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self.model(self.g, self.features)
        val_acc = self.evaluator(logits[self.val_x], self.labels[self.val_x])
        test_acc = self.evaluator(logits[self.test_x], self.labels[self.test_x])
        return val_acc, test_acc
    
class GSR(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, params):
        super(GSR, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device, params)
        self.update_configs()

    def update_configs(self):
        self.config.model = 'GSR'
        DWConfig_dict = {
            'cora': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'citeseer': {
                'num_walks': 10,
                'window_size': 13,
                'walk_length': 100,
            },
            'blogcatalog': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'flickr': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'arxiv': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'airport': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
        }

        DEConfig_dict = {
            'cora': {
                'k': 'full'
            },
            'citeseer': {
                'k': 6
            },
        }

        if self.config.semb == 'dw':
            self.config.se_num_walks = DWConfig_dict[self.config.dataset]['num_walks']
            self.config.se_window_size = DWConfig_dict[self.config.dataset]['window_size']
            self.config.se_walk_length = DWConfig_dict[self.config.dataset]['walk_length']
            self.config.structural_em_file = f'{TEMP_PATH}{self.config.model}/structural_embs/{self.config.dataset}/{self.config.dataset}_nw{self.config.se_num_walks}_wl{self.config.se_walk_length}_ws{self.config.se_window_size}.{self.config.semb}_emb'
            self.config.pretrain_conf = f"_lr{self.config.prt_lr}_bsz{self.config.p_batch_size}_pi{self.config.p_epochs}_enc{self.config.gnn_model}"
            f"_dec-l{self.config.decoder_layer}_hidden{self.config.decoder_n_hidden}-prt_intra_w-{self.config.intra_weight}_ncek{self.config.nce_k}_fanout{self.config.fan_out}_prdo{self.config.pre_dropout}_act_{self.config.activation}_d{self.config.n_hidden}_pss{self.config.p_schedule_step}"
        if self.config.semb == 'de':
            self.config.se_k = DEConfig_dict[self.config.dataset]['k']
            self.config.structural_em_file = f'{TEMP_PATH}{self.config.model}/structural_embs/{self.config.dataset}/{self.config.dataset}_k{self.config.se_k}.{self.config.semb}_emb'
            self.config.pretrain_conf = f"_lr{self.config.prt_lr}_bsz{self.config.p_batch_size}_pi{self.config.p_epochs}_enc{self.config.gnn_model}"
            f"_dec-l{self.config.decoder_layer}_hidden{self.config.decoder_n_hidden}-prt_intra_w-{self.config.intra_weight}_ncek{self.config.nce_k}_fanout{self.config.fan_out}_prdo{self.config.pre_dropout}_act_{self.config.activation}_d{self.config.n_hidden}_pdl{self.config.poly_decay_lr}_sek{self.confg.se_k}"
        self.config.model_conf = self.config
        self.config.pretrain_model_ckpt = f"{TEMP_PATH}{self.config.model}/p_model_ckpts/{self.config.dataset}/{self.config.pretrain_conf}.ckpt"
        
    
    
    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        if adj.is_sparse:
            indices = adj.coalesce().indices()
            values = adj.coalesce().values()
            shape = adj.coalesce().shape
            num_nodes = features.shape[0]
            loop_edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(adj.device)
            loop_edge_index = torch.cat([indices, loop_edge_index], dim=1)
            loop_values = torch.ones(num_nodes).to(adj.device)
            loop_values = torch.cat([values, loop_values], dim=0)
            adj = torch.sparse_coo_tensor(indices=loop_edge_index, values=loop_values, size=shape)
        else:
            adj += torch.eye(adj.shape[0]).to(self.device)
            adj = adj.to_sparse()
        edge = adj.to_dense().nonzero().t().cpu()
        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)
        
        feat = {'F': features, 'S': get_structural_feature(g, self.config)}
        self.config.feat_dim = {v: feat.shape[1] for v, feat in feat.items()}
        supervision = edict({'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask, 'labels': labels})
        print(f'{self.config}\nStart training..')
        p_model = GSR_pretrain(g, self.config).to(self.config.device)
        # print(p_model)
        # ! Train Phase 1: Pretrain
        if self.config.p_epochs > 0:
            # os.remove(self.config.pretrain_model_ckpt)  # Debug Only
            if os.path.exists(self.config.pretrain_model_ckpt):

                p_model.load_state_dict(torch.load(self.config.pretrain_model_ckpt, map_location=self.config.device))
                print(f'Pretrain embedding loaded from {self.config.pretrain_model_ckpt}')
            else:
                print('>>>> PHASE 1 - Pretraining and Refining Graph Structure <<<<<')
                views = ['F', 'S']
                optimizer = torch.optim.Adam(
                    p_model.parameters(), lr=self.config.prt_lr, weight_decay=self.config.weight_decay)
                if self.config.p_schedule_step > 1:
                    scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=self.config.p_schedule_step,
                                                                end_learning_rate=0.0001, power=2.0)
                # Construct virtual relation triples
                p_model_ema = GSR_pretrain(g, self.config).to(self.config.device)
                moment_update(p_model, p_model_ema, 0)  # Copy
                moco_memories = {v: MemoryMoCo(self.config.n_hidden, self.config.nce_k,  # Single-view contrast
                                            self.config.nce_t, device=self.config.device).to(self.config.device)
                                for v in views}
                criterion = NCESoftmaxLoss(self.config.device)
                pretrain_loader = get_pretrain_loader(g.cpu(), self.config)

                for epoch_id in range(self.config.p_epochs):
                    for step, (input_nodes, edge_subgraph, blocks) in enumerate(pretrain_loader):
                        t0 = time()
                        blocks = [b.to(self.config.device) for b in blocks]
                        edge_subgraph = edge_subgraph.to(self.config.device)
                        input_feature = {v: feat[v][input_nodes].to(self.config.device) for v in views}

                        # ===================Moco forward=====================
                        p_model.train()

                        q_emb = p_model(edge_subgraph, blocks, input_feature, mode='q')
                        std_dict = {v: round(q_emb[v].std(dim=0).mean().item(), 4) for v in ['F', 'S']}
                        print(f"Std: {std_dict}")

                        if std_dict['F'] == 0 or std_dict['S'] == 0:
                            print(f'\n\n????!!!! Same Embedding Epoch={epoch_id}Step={step}\n\n')
                            # q_emb = p_model(edge_subgraph, blocks, input_feature, mode='q')

                        with torch.no_grad():
                            k_emb = p_model_ema(edge_subgraph, blocks, input_feature, mode='k')
                        intra_out, inter_out = [], []

                        for tgt_view, memory in moco_memories.items():
                            for src_view in views:
                                if src_view == tgt_view:
                                    intra_out.append(memory(
                                        q_emb[f'{tgt_view}'], k_emb[f'{tgt_view}']))
                                else:
                                    inter_out.append(memory(
                                        q_emb[f'{src_view}->{tgt_view}'], k_emb[f'{tgt_view}']))

                        # ===================backward=====================
                        # ! Self-Supervised Learning
                        intra_loss = torch.stack([criterion(out_) for out_ in intra_out]).mean()
                        inter_loss = torch.stack([criterion(out_) for out_ in inter_out]).mean()
                        # ! Loss Fusion
                        loss_tensor = torch.stack([intra_loss, inter_loss])
                        intra_w = float(self.config.intra_weight)
                        loss_weights = torch.tensor([intra_w, 1 - intra_w], device=self.config.device)
                        loss = torch.dot(loss_weights, loss_tensor)
                        # ! Semi-Supervised Learning
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        moment_update(p_model, p_model_ema, self.config.momentum_factor)
                        print_log({'Epoch': epoch_id, 'Batch': step, 'Time': time() - t0,
                                'intra_loss': intra_loss.item(), 'inter_loss': inter_loss.item(),
                                'overall_loss': loss.item()})

                        if self.config.p_schedule_step > 1:
                            scheduler_poly_lr_decay.step()

                    epochs_to_save = P_EPOCHS_SAVE_LIST + ([1, 2, 3, 4] if self.config.dataset == 'arxiv' else [])
                    if epoch_id + 1 in epochs_to_save:
                        # Convert from p_epochs to current p_epoch checkpoint
                        ckpt_name = self.config.pretrain_model_ckpt.replace(f'_pi{self.config.p_epochs}', f'_pi{epoch_id + 1}')
                        torch.save(p_model.state_dict(), ckpt_name)
                        print(f'Model checkpoint {ckpt_name} saved.')

                torch.save(p_model.state_dict(), self.config.pretrain_model_ckpt)

        # ! Train Phase 2: Graph Structure Refine
        print('>>>> PHASE 2 - Graph Structure Refine <<<<< ')

        if self.config.p_epochs <= 0 or self.config.add_ratio + self.config.rm_ratio == 0:
            print('Use original graph!')
            g_new = g
        else:
            if os.path.exists(self.config.refined_graph_file):
                print(f'Refined graph loaded from {self.config.refined_graph_file}')
                g_new = dgl.load_graphs(self.config.refined_graph_file)[0][0]
            else:
                g_new = p_model.refine_graph(g, feat)
                dgl.save_graphs(self.config.refined_graph_file, [g_new])

        # ! Train Phase 3:  Node Classification
        f_model = GSR_finetune(self.config).to(self.config.device)
        print(f_model)
        # Copy parameters
        if self.config.p_epochs > 0:
            para_copy(f_model, p_model.encoder.F, paras_to_copy=['conv1.weight', 'conv1.bias'])
        optimizer = torch.optim.Adam(f_model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        stopper = EarlyStopping(patience=self.config.early_stop, path=self.config.checkpoint_file) if self.config.early_stop else None
        del g, feat, p_model
        torch.cuda.empty_cache()

        print('>>>> PHASE 3 - Node Classification <<<<< ')
        trainer_func = FullBatchTrainer
        trainer = trainer_func(model=f_model, g=g_new, features=features, sup=supervision, config=self.config,
                            stopper=stopper, optimizer=optimizer, loss_func=torch.nn.CrossEntropyLoss())
        trainer.run()
        trainer.eval_and_save()

        return self.config
            
def scalable_graph_refine(g, emb, rm_num, add_num, batch_size, fsim_weight, device, norm=False):
    def _update_topk(sim, start, mask, k, prev_inds, prev_sim, largest):
        # Update TopK similarity and inds
        top_inds, top_sims = topk_sim_edges(sim + mask, k, start, largest)
        temp_inds = torch.cat((prev_inds, top_inds))
        temp_sim = torch.cat((prev_sim, top_sims))
        current_best = temp_sim.topk(k, largest=largest).indices
        return temp_sim[current_best], temp_inds[current_best]

    edges = set(graph_edge_to_lot(g))
    num_batches = int(g.num_nodes() / batch_size) + 1
    if add_num + rm_num == 0:
        return g.edges()

    if norm:
        # Since maximum value of a similarity matrix is fixed as 1, we only have to calculate the minimum value
        fsim_min, ssim_min = 99, 99
        for row_i in tqdm(range(num_batches), desc='Calculating minimum similarity'):
            # ! Initialize batch inds
            start = row_i * batch_size
            end = min((row_i + 1) * batch_size, g.num_nodes())
            if end <= start:
                break

            # ! Calculate similarity matrix
            fsim_min = min(fsim_min, cosine_sim_torch(emb['F'][start:end], emb['F']).min())
            ssim_min = min(ssim_min, cosine_sim_torch(emb['S'][start:end], emb['S']).min())
    # ! Init index and similairty tensor
    # Edge indexes should not be saved as floats in triples, since the number of nodes may well exceeds the maximum of float16 (65504)
    rm_inds, add_inds = [torch.tensor([(0, 0) for i in range(_)]).type(torch.int32).to(device)
                         for _ in [1, 1]]  # Init with one random point (0, 0)
    add_sim = torch.ones(1).type(torch.float16).to(device) * -99
    rm_sim = torch.ones(1).type(torch.float16).to(device) * 99

    for row_i in tqdm(range(num_batches), desc='Batch filtering edges'):
        # ! Initialize batch inds
        start = row_i * batch_size
        end = min((row_i + 1) * batch_size, g.num_nodes())
        if end <= start:
            break

        # ! Calculate similarity matrix
        f_sim = cosine_sim_torch(emb['F'][start:end], emb['F'])
        s_sim = cosine_sim_torch(emb['S'][start:end], emb['S'])
        if norm:
            f_sim = (f_sim - fsim_min) / (1 - fsim_min)
            s_sim = (s_sim - ssim_min) / (1 - ssim_min)
        sim = fsim_weight * f_sim + (1 - fsim_weight) * s_sim

        # ! Get masks
        # Edge mask
        edge_mask, diag_mask = [torch.zeros_like(sim).type(torch.int8) for _ in range(2)]
        row_gids, col_ids = g.out_edges(g.nodes()[start: end])
        edge_mask[row_gids - start, col_ids] = 1
        # Diag mask
        diag_r, diag_c = zip(*[(_ - start, _) for _ in range(start, end)])
        diag_mask[diag_r, diag_c] = 1
        # Add masks: Existing edges and diag edges should be masked
        add_mask = (edge_mask + diag_mask) * -99
        # Remove masks: Non-Existing edges should be masked (diag edges have 1 which is maximum value)
        rm_mask = (1 - edge_mask) * 99

        # ! Update edges to remove and add
        if rm_num > 0:
            k = max(len(rm_sim), rm_num)
            rm_sim, rm_inds = _update_topk(sim, start, rm_mask, k, rm_inds, rm_sim, largest=False)
        if add_num > 0:
            k = max(len(add_sim), add_num)
            add_sim, add_inds = _update_topk(sim, start, add_mask, k, add_inds, add_sim, largest=True)

    # ! Graph refinement
    if rm_num > 0:
        rm_edges = [tuple(_) for _ in rm_inds.cpu().numpy().astype(int).tolist()]
        edges -= set(rm_edges)
    if add_num > 0:
        add_edges = [tuple(_) for _ in add_inds.cpu().numpy().astype(int).tolist()]
        edges |= set(add_edges)
    # assert uf.load_pickle('EdgesGeneratedByOriImplementation') == sorted(edges)
    return edges


def topk_sim_edges(sim_mat, k, row_start_id, largest):
    v, i = torch.topk(sim_mat.flatten(), k, largest=largest)
    inds = np.array(np.unravel_index(i.cpu().numpy(), sim_mat.shape)).T
    inds[:, 0] = inds[:, 0] + row_start_id
    ind_tensor = torch.tensor(inds).to(sim_mat.device)
    # ret = torch.cat((torch.tensor(inds).to(sim_mat.device), v.view((-1, 1))), dim=1)
    return ind_tensor, v  # v.view((-1, 1)


RES_PATH = 'temp_results/'
SUM_PATH = 'results/'
LOG_PATH = 'log/'
TEMP_PATH = 'temp/'

class ModelConfig(metaclass=ABCMeta):
    """

    """

    def __init__(self, model):
        self.model = model
        self.exp_name = 'default'
        self.seed = 0
        self.birth_time = get_cur_time(t_format='%m_%d-%H_%M_%S')
        # Other attributes
        self._model_conf_list = None
        self._interested_conf_list = ['model']
        self._file_conf_list = ['checkpoint_file', 'res_file']

    def __str__(self):
        # Print all attributes including data and other path settings added to the config object.
        return str({k: v for k, v in self.model_conf.items() if k != '_interested_conf_list'})

    @property
    @abstractmethod
    def f_prefix(self):
        # Model config to str
        return ValueError('The model config file name must be defined')

    @property
    @abstractmethod
    def checkpoint_file(self):
        # Model config to str
        return ValueError('The checkpoint file name must be defined')

    @property
    def res_file(self):
        return f'{RES_PATH}{self.model}/{self.dataset}/l{self.train_percentage:02d}/{self.f_prefix}.txt'
        # return f'{RES_PATH}{self.model}/{self.dataset}/{self.f_prefix}.txt'

    @property
    def model_conf(self):
        # Print the model settings only.
        return {k: self.__dict__[k] for k in self._model_conf_list}

    def get_sub_conf(self, sub_conf_list):
        # Generate subconfig dict using sub_conf_list
        return {k: self.__dict__[k] for k in sub_conf_list}

    def update__model_conf_list(self, new_conf=[]):
        # Maintain a list of interested configs
        other_configs = ['_model_conf_list', '_file_conf_list']
        if len(new_conf) == 0:  # initialization
            self._model_conf_list = sorted(list(self.__dict__.copy().keys()))
            for uninterested_config in other_configs:
                self._model_conf_list.remove(uninterested_config)
        else:
            self._model_conf_list = sorted(self._model_conf_list + new_conf)

    def update_modified_conf(self, conf_dict):
        self.__dict__.update(conf_dict)
        self._interested_conf_list += list(conf_dict)
        unwanted_items = ['log_on', 'gpu', 'train_phase', 'num_workers']
        for item in unwanted_items:
            if item in self._interested_conf_list:
                self._interested_conf_list.remove(item)
        mkdir_list([getattr(self, _) for _ in self._file_conf_list])


class SEConfig(ModelConfig):

    def __init__(self, args):
        super(SEConfig, self).__init__('StructureEmbedding')

        # # ! Model setting
        SEConfig_dict = {
            'cora': {
                'num_walks': 100,
                'window_size': 11,
                'walk_length': 100,
            },
            'citeseer': {
                'num_walks': 10,
                'window_size': 13,
                'walk_length': 100,
            },
            'pubmed': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'arxiv': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
            'airport': {
                'num_walks': 12,
                'window_size': 11,
                'walk_length': 100,
            },
        }

        self.dataset = args.dataset
        # dataset_config = SEConfig_dict[self.dataset]
        # self.se_num_walks = dataset_config['num_walks']
        # self.se_walk_length = dataset_config['walk_length']
        # self.se_window_size = dataset_config['window_size']
        self.se_num_walks = args.se_num_walks
        self.se_walk_length = args.se_walk_length
        self.se_window_size = args.se_window_size
        self.se_n_hidden = 64
        self.se_num_workers = 32
        self.train_percentage = args.train_percentage

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_{self.model}_nw{self.se_num_walks}_wl{self.se_walk_length}_ws{self.se_window_size}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}.ckpt"



