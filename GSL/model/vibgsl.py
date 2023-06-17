import time
import math
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import Linear, ReLU, Dropout
from torch.autograd import Variable
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.glob import AvgPooling
from GSL.model import BaseModel
from GSL.utils import *
from GSL.learner import *
from GSL.encoder import *
from GSL.metric import *
from GSL.processor import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INF = 1e20
VERY_SMALL_NUMBER = 1e-12


class VIBGSL(BaseModel):
    def __init__(self, device, args, num_node_features, num_classes):
        super(VIBGSL, self).__init__(device)
        self.args = args
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.backbone = args.backbone
        self.hidden_dim = args.hidden_dim
        self.IB_size = args.IB_size
        self.graph_metric_type = args.graph_metric_type
        self.graph_type = args.graph_type
        self.top_k = args.top_k
        self.epsilon = args.epsilon
        self.beta = args.beta
        self.num_per = args.num_per

        if self.backbone == "GCN":
            self.backbone_gnn = GCN_dgl(in_channels=self.num_node_features, hidden_channels=self.hidden_dim,
                                        out_channels=self.IB_size * 2, num_layers=2, dropout=0., dropout_adj=0.)
        elif self.backbone == "GIN":
            self.backbone_gnn = GIN(input_dim=self.num_node_features, hidden_dim=self.hidden_dim,
                                    output_dim=self.IB_size*2, num_layers=2)
        elif self.backbone == "GAT":
            self.backbone_gnn = GAT_dgl(nfeat=self.num_node_features, nhid=self.hidden_dim, nclass=self.IB_size * 2,
                                        dropout=0.5, alpha=0.2, nheads=4)

        self.pool = AvgPooling()

        metric = InnerProductSimilarity()
        processors = [ProbabilitySparsify(temperature=0.05)]
        if self.args.graph_include_self:
            processors.append(AddEye())
        self.graph_learner = MLPLearner(metric, processors, nlayers=2, in_dim=self.num_node_features,
                                        hidden_dim=self.hidden_dim, activation=torch.relu, sparse=False)

        self.classifier = torch.nn.Sequential(Linear(self.IB_size, self.IB_size), ReLU(), Dropout(p=0.5),
                                              Linear(self.IB_size, self.num_classes))

        if torch.cuda.is_available():
            self.backbone_gnn = self.backbone_gnn.cuda()
            self.graph_learner = self.graph_learner.cuda()
            self.classifier = self.classifier.cuda()

    def __repr__(self):
        return self.__class__.__name__

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def to(self, device):
        self.backbone_gnn.to(device)
        self.graph_learner.to(device)
        self.classifier.to(device)
        return self

    def reset_parameters(self):
        self.backbone_gnn.reset_parameters()
        self.graph_learner.param_init()
        for module in self.classifier:
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()

    def reparametrize_n(self, mu, std, n=1):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def forward(self, graphs):
        graphs_list = []
        new_graphs_list = []

        num_sample = graphs.batch_size
        for i in range(num_sample):
            graph = dgl.slice_batch(graphs, i)
            graphs_list.append(graph)
            feat = graph.ndata.pop('feat')
            new_adj = self.graph_learner(feat)

            # construct new graph
            src, dst = torch.nonzero(new_adj, as_tuple=True)
            weights = new_adj[src, dst]  # get weights of edges
            new_graph = dgl.graph((src, dst))
            new_graph.ndata['feat'] = feat
            new_graph.edata['w'] = weights
            new_graphs_list.append(new_graph)
        new_batch = dgl.batch(new_graphs_list)

        feat = new_batch.ndata.pop('feat')
        node_embs = self.backbone_gnn(feat, new_batch)
        graph_embs = self.pool(new_batch, node_embs)

        mu = graph_embs[:, :self.IB_size]
        std = F.softplus(graph_embs[:, self.IB_size:]-self.IB_size, beta=1)
        new_graph_embs = self.reparametrize_n(mu, std, num_sample)

        logits = self.classifier(new_graph_embs)

        return (mu, std), logits, graphs_list, new_graphs_list

    def fit(self, dataset, folds, epochs, batch_size, test_batch_size, lr,
                                  lr_decay_factor, lr_decay_step_size, weight_decay, logger=None):
        val_losses, val_accs, test_accs, durations = [], [], [], []
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds, self.args.seed))):

            fold_val_losses = []
            fold_val_accs = []
            fold_test_accs = []

            infos = dict()

            train_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(train_idx), batch_size=batch_size)
            val_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(val_idx), batch_size=test_batch_size)
            test_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(test_idx), batch_size=test_batch_size)

            self.to(device).reset_parameters()
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=eval(weight_decay))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            for epoch in range(1, epochs + 1):
                train_cls_loss, train_KL_loss, train_loss, train_acc = train_VGIB(self, optimizer, train_loader)
                val_cls_loss, val_KL_loss, val_loss, val_acc = eval_VGIB_loss(self, val_loader)
                val_losses.append(val_loss)
                fold_val_losses.append(val_loss)
                fold_val_accs.append(val_acc)
                val_accs.append(val_acc)

                test_acc, data, graphs_list, new_graphs_list, pred_y = eval_VGIB_acc(self, test_loader)
                test_accs.append(test_acc)
                fold_test_accs.append(test_acc)
                eval_info = {
                    'fold': fold,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_cls_loss': val_cls_loss,
                    "val_KL_loss": val_KL_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                }
                infos[epoch] = eval_info

                if logger is not None:
                    logger(eval_info)

                if epoch % lr_decay_step_size == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_decay_factor * param_group['lr']
                if epoch % 1 == 0:
                    print(
                        'Epoch: {:d}, train loss: {:.3f}, train acc: {:.3f}, val loss: {:.5f}, val acc: {:.3f}, test scc: {:.3f}'
                        .format(epoch, eval_info["train_loss"], eval_info["train_acc"], eval_info["val_loss"],
                                eval_info["val_acc"], eval_info["test_acc"]))

            fold_val_loss, argmin = tensor(fold_val_losses).min(dim=0)
            fold_val_acc, argmax = tensor(fold_val_accs).max(dim=0)
            fold_test_acc = fold_test_accs[argmin]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            durations.append(t_end - t_start)
            print('Fold: {:d}, train acc: {:.3f}, Val loss: {:.3f}, Val acc: {:.5f}, Test acc: {:.3f}'
                  .format(eval_info["fold"], eval_info["train_acc"], fold_val_loss, fold_val_acc, fold_test_acc))

        val_losses, val_accs, test_accs, duration = tensor(val_losses), tensor(val_accs), tensor(test_accs), tensor(
            durations)
        val_losses, val_accs, test_accs = val_losses.view(folds, epochs), val_accs.view(folds, epochs), test_accs.view(
            folds, epochs)

        min_val_loss, argmin = val_losses.min(dim=1)
        test_acc = test_accs[torch.arange(folds, dtype=torch.long), argmin]

        val_loss_mean = min_val_loss.mean().item()
        test_acc_mean = test_acc.mean().item()
        test_acc_std = test_acc.std().item()
        duration_mean = duration.mean().item()
        print(test_acc)
        print('Val Loss: {:.4f}, Test Accuracy: {:.3f}+{:.3f}, Duration: {:.3f}'
              .format(val_loss_mean, test_acc_mean, test_acc_std, duration_mean))

        return test_acc, test_acc_mean, test_acc_std


def k_fold(dataset, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    if isinstance(dataset, DGLDataset):
        labels = dataset.graph_labels
        for _, idx in skf.split(torch.zeros(len(dataset)), labels):
            test_indices.append(torch.from_numpy(idx).to(torch.long))
    elif isinstance(dataset, list):
        for _, idx in skf.split(torch.zeros(len(dataset)), [data.y for data in dataset]):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def num_graphs(graphs):
    return graphs.batch_size


def train_VGIB(model, optimizer, loader):
    model.train()

    total_loss = 0
    total_class_loss = 0
    total_KL_loss = 0
    correct = 0
    for graphs, labels in loader:
        optimizer.zero_grad()
        graphs = graphs.to(device)
        labels = labels.reshape(-1).to(device)
        (mu, std), logits, _, _ = model(graphs)
        class_loss = F.cross_entropy(logits, labels).div(math.log(2))
        KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))

        loss = class_loss + model.beta * KL_loss
        loss.backward()
        total_loss += loss.item() * num_graphs(graphs)
        total_class_loss += class_loss.item() * num_graphs(graphs)
        total_KL_loss += KL_loss.item() * num_graphs(graphs)
        optimizer.step()
        pred = logits.max(1)[1]
        correct += pred.eq(labels.view(-1)).sum().item()
    return total_class_loss / len(loader.sampler), total_KL_loss / len(loader.sampler), total_loss / len(loader.sampler), correct / len(loader.sampler)


def eval_VGIB_acc(model, loader):
    model.eval()

    correct = 0
    graphs_list = []
    new_graphs_list = []
    for graphs, labels in loader:
        graphs = graphs.to(device)
        labels = labels.reshape(-1).to(device)
        with torch.no_grad():
            _, logits, tmp_graphs_list, tmp_new_graphs_list = model(graphs)
            graphs_list += tmp_graphs_list
            new_graphs_list += tmp_new_graphs_list
            pred = logits.max(1)[1]
        correct += pred.eq(labels.view(-1)).sum().item()
    return correct / len(loader.sampler), graphs, graphs_list, new_graphs_list, pred


def eval_VGIB_loss(model, loader):
    model.eval()
    total_loss = 0
    total_class_loss = 0
    total_KL_loss = 0
    correct = 0
    for graphs, labels in loader:
        graphs = graphs.to(device)
        labels = labels.reshape(-1).to(device)
        with torch.no_grad():
            (mu, std), logits, _, _ = model(graphs)
            pred = logits.max(1)[1]
        correct += pred.eq(labels.view(-1)).sum().item()
        class_loss = F.cross_entropy(logits, labels).div(math.log(2))
        KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        loss = class_loss.item() + model.beta * KL_loss.item()
        total_loss += loss * num_graphs(graphs)
        total_class_loss += class_loss * num_graphs(graphs)
        total_KL_loss += KL_loss * num_graphs(graphs)
    return total_class_loss.item() / len(loader.sampler), total_KL_loss.item() / len(loader.sampler), total_loss / len(loader.sampler), correct / len(loader.sampler)