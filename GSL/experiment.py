import os
import pickle
from contextlib import redirect_stdout

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
# from hyperopt import Trials, fmin, hp, tpe

from GSL.data import *
from GSL.encoder import *
from GSL.model import *
from GSL.utils import (accuracy, get_knn_graph, macro_f1, micro_f1,
                       random_add_edge, random_drop_edge, seed_everything,
                       sparse_mx_to_torch_sparse_tensor)


class Experiment(object):
    def __init__(self, model_name, 
                 dataset, ntrail, 
                 data_path, config_path,
                 sparse: bool = False,
                 metric: str = 'acc',
                 gpu_num: int = 0,
                 use_knn: bool = False,
                 k: int = 5,
                 drop_rate: float = 0.,
                 add_rate: float = 0.,
                 use_mettack: bool = False,
                 ptb_rate: float = 0.05
                 ):
        
        self.sparse = sparse
        self.ntrail = ntrail
        self.metric = metric
        self.config_path = config_path
        self.device = torch.device("cuda:"+str(gpu_num) if torch.cuda.is_available() else "cpu")

        self.model_name = model_name
        self.dataset_name = dataset.lower()
        self.model_dict = {
            'GCN': GCN_Trainer, 'MLP': MLP_Trainer, 'GRCN': GRCN, 'ProGNN': ProGNN, 'IDGL': IDGL, 'GEN': GEN, 'CoGSL': CoGSL, 
            'SLAPS': SLAPS, 'SUBLIME': SUBLIME, 'STABLE': STABLE, 'GTN': GTN, 'HGSL': HGSL, 
            'HGPSL': HGPSL, 'VIBGSL': VIBGSL, 'HESGSL': HESGSL
            # 'nodeformer': NodeFormer,
        }

        # Load graph datasets
        if self.dataset_name in ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv',
                                 'cornell', 'texas', 'wisconsin', 'actor']:
            self.data = Dataset(root=data_path, name=self.dataset_name, use_mettack=use_mettack, ptb_rate=ptb_rate)
        elif self.dataset_name in ['acm', 'dblp', 'yelp']:
            self.data = HeteroDataset(root=data_path, name=self.dataset_name)
        elif self.dataset_name in ['imdb-b', 'imdb-m', 'collab', 'reddit-b', 'mutag',
                                   'proteins', 'peptides-func', 'peptides-struct']:
            self.data = GraphDataset(root=data_path, name=self.dataset_name)

        if isinstance(self.data, Dataset):
            # Modify graph structures
            adj = self.data.adj
            features = self.data.features

            # mask = feature_mask(features, 0.8)
            # apply_feature_mask(features, mask)

            # Randomly drop edges
            if drop_rate > 0:
                adj = random_drop_edge(adj, drop_rate)

            # Randomly add edges
            if add_rate > 0:
                adj = random_add_edge(adj, drop_rate)

            # Use knn graph instead of the original structure
            if use_knn:
                adj = get_knn_graph(features, k, self.dataset_name)

            # Sparse or notyao
            if not self.sparse:
                self.data.adj = torch.from_numpy(adj.todense()).to(torch.float)
            else:
                self.data.adj = sparse_mx_to_torch_sparse_tensor(adj)

        # Select the metric of evaluation
        self.eval_metric = {
            'acc': accuracy,
            'macro-f1': macro_f1,
            'micro-f1': micro_f1,
        }[metric]
        
    def run(self, params=None):
        """
        Run the experiment
        """
        test_results = []
        num_feat, num_class = self.data.num_feat, self.data.num_class

        for i in range(self.ntrail):

            seed_everything(i)

            # Initialize the GSL model
            if self.model_name in ['SLAPS', 'CoGSL', 'HGSL']:
                model = self.model_dict[self.model_name](num_feat, num_class, self.eval_metric,
                                                         self.config_path, self.dataset_name, self.device, self.data) # TODO modify the config according to the search space
            else:
                model = self.model_dict[self.model_name](num_feat, num_class, self.eval_metric,
                                                         self.config_path, self.dataset_name, self.device, params)
            self.model = model.to(self.device)
            if isinstance(self.data, GraphDataset):
                self.data = self.data.dgl_dataset
            else:
                self.data = self.data.to(self.device)

            # Structure Learning
            self.model.fit(self.data)

            result = self.model.best_result
            test_results.append(result)
            print('Run: {} | Test result: {}'.format(i+1, result))

        # TODO: support multiple metrics
        exp_info = '------------------------------------------------------------------------------\n' \
                   'Experimental settings: \n' \
                   'Model: {} | Dataset: {} | Metric: {} \n' \
                   'Result: {} \n' \
                   'Final test result: {} +- {} \n' \
                   '------------------------------------------------------------------------------'.\
                   format(self.model_name, self.dataset_name, self.metric,
                          test_results, np.mean(test_results), np.std(test_results))
        print(exp_info)
        return np.mean(test_results)
    
    def objective(self, params):
        with redirect_stdout(open(os.devnull, "w")):
            return {'loss': -self.run(params), 'status': 'ok'}

    def hp_search(self, hyperspace_path):
        with open(hyperspace_path, "r") as fin:
            raw_text = fin.read()
            raw_space = edict(yaml.safe_load(raw_text))
        
        space_hyperopt = {} 
        for key, config in raw_space.items():
            if config.type == 'choice':
                space_hyperopt[key] = hp.choice(key, config.values)
            
            elif config.type == 'uniform':
                space_hyperopt[key] = hp.uniform(key, config.bounds[0], config.bounds[1])
            
            # 可以扩展其他类型  
        print(space_hyperopt)
        trials = Trials()
        best = fmin(self.objective, space_hyperopt, algo=tpe.suggest, max_evals=100, trials=trials)
        print(trials.best_trial)
        
        with open('trails.pkl', 'wb') as f:
            pickle.dump(trials, f)
        