import easydict
import numpy as np
import torch.nn as nn
import yaml

from GSL.learner import *


class BaseModel(nn.Module):
    '''
    Abstract base class for graph structure learning models.
    '''
    def __init__(self, num_feat, num_class, metric, config_path, dataset, device, params=None):
        super(BaseModel, self).__init__()
        self.num_feat = num_feat
        self.num_class = num_class
        self.metric = metric
        self.device = device
        self.load_config(config_path, dataset, params)

    def check_adj(self, adj):
        """
        Check if the modified adjacency is symmetric and unweighted.
        """
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.tocsr().max() == 1, "Max value should be 1!"
        assert adj.tocsr().min() == 0, "Min value should be 0!"

    def load_config(self, config_path, dataset, params):
        """
        Load the hyper-parameters required for the models.
        """
        with open(config_path, "r") as fin:
            raw_text = fin.read()
            configs = [easydict.EasyDict(yaml.safe_load(raw_text))]

        all_config = configs[0]
        config = all_config.Default
        dataset_config = all_config.get(dataset, None)
        if dataset_config is not None:
            config.update(dataset_config)
        self.config = config
        self.load_hypers(params)

    def load_hypers(self, params):
        if params is None:
            return
        for key, value in params.items():
            self.config[key] = value
