from .gat import GAT, GAT_dgl
from .gcn import GCN, GCNConv, GCNConv_dgl, SparseDropout, GraphEncoder, GCNConv_diag, GCN_dgl
from .gin import GIN
from .graphsage import GraphSAGE
from .mlp import MLP
from .nodeformer_conv import NodeFormerConv

__all__ = [
    "GAT",
    "GAT_dgl",
    "GCN",
    "GCNConv",
    "GCNConv_dgl",
    "SparseDropout",
    "GraphEncoder",
    "GCNConv_diag",
    "GCN_dgl",
    "GIN",
    "GraphSAGE",
    "MLP",
    "NodeFormerConv"
]

classes = __all__