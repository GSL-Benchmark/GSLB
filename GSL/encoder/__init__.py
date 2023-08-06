from .gat import GAT, GAT_dgl
from .gcn import GCN, GCNConv, GCNConv_dgl, SparseDropout, GraphEncoder, GCNConv_diag, GCN_dgl
from .gin import GIN
from .graphsage import GraphSAGE
from .mlp import MLP
from .nodeformer_conv import NodeFormerConv
from .gprgnn import GPRGNN, GPR_prop

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
    "NodeFormerConv",
    "GPR_prop",
    "GPRGNN"
]

classes = __all__