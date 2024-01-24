import os.path as osp
import numpy as np
import dgl
import torch
import torch.nn.functional as F
from dgl.data import TUDataset


class GraphDataset:
    def __init__(self, root, name):
        self.root = osp.expanduser(osp.normpath(root))
        dict_dataset_name = {
            'imdb-b': 'IMDB-BINARY',
            'imdb-m': 'IMDB-MULTI',
            'collab': 'COLLAB',
            'reddit-b': 'REDDIT-BINARY',
        }
        self.name = dict_dataset_name[name]

        assert self.name in [
            "IMDB-BINARY", "REDDIT-BINARY", "COLLAB", "IMDB-MULTI",
        ], (
            "Currently only support IMDB-BINARY, REDDIT-BINARY, COLLAB, IMDB-MULTI"
        )

        self.dgl_dataset = self.load_data()
        self.graphs, self.labels = zip(*[self.dgl_dataset[i] for i in range(len(self.dgl_dataset))])
        self.labels = torch.tensor(self.labels)
        self.num_feat = self.dgl_dataset[0][0].ndata['feat'].shape[1]
        self.num_class = self.dgl_dataset.num_labels

    def __len__(self):
        return len(self.dgl_dataset)

    def load_data(self):
        # TODO: to_simple will be conducted before every access, which results in
        #  the computed features being overwritten. Maybe we could abstract this
        #  function to a dgl.transform class.
        dataset = TUDataset(self.name, raw_dir=self.root)
        # dataset = TUDataset(self.name, raw_dir=self.root, transform=dgl.to_simple)
        graph, _ = dataset[0]

        if "feat" not in graph.ndata:
            max_degree = 0
            degs = []
            for g, _ in dataset:
                degs += [g.in_degrees().to(torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                # use one-hot degree embedding as node features
                for g, _ in dataset:
                    deg = g.in_degrees()
                    g.ndata['feat'] = F.one_hot(deg, num_classes=max_degree+1).to(torch.float)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                for g, _ in dataset:
                    deg = g.in_degrees().to(torch.float)
                    deg = (deg - mean) / std
                    g.ndata['feat'] = deg.view(-1, 1)

        return dataset



if __name__ == "__main__":

    # graph classification dataset
    data_path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = GraphDataset(root=data_path, name="IMDB-BINARY")
