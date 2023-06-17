from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import AvgPooling, GraphConv, MaxPooling, GATConv, SAGEConv, GINConv
import dgl.function as fn
from dgl.base import ALL, is_all
from dgl.dataloading import GraphDataLoader
from dgl import DGLGraph
from dgl._sparse_ops import _gsddmm, _gspmm
from dgl.ops import edge_softmax
from dgl.backend import astype
from dgl.heterograph_index import HeteroGraphIndex
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Function
import scipy
from GSL.model import BaseModel
from GSL.utils import *
from GSL.learner import *
from GSL.encoder import *
from GSL.metric import *
from GSL.processor import *


class HGPSL(BaseModel):
    """
    Hierarchical Graph Pooling with Structure Learning (AAAI 2020)
    The most of code implementation comes from: https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgp_sl
    """
    def __init__(self, device, config, num_node_features, num_classes):
        super(HGPSL, self).__init__(device)
        self.folds = config.folds
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.in_dim = num_node_features
        self.out_dim = num_classes
        self.pool_ratio = config.pool_ratio
        self.sample = config.sample
        self.sparse_attn = config.sparse_attn
        self.sl = config.sl
        self.lamb = config.lamb
        self.patience = config.patience
        self.dropout = config.dropout

        convpools = []
        for i in range(self.num_layers):
            c_in = self.in_dim if i == 0 else self.hidden_dim
            c_out = self.hidden_dim
            use_pool = i != self.num_layers - 1
            convpools.append(
                ConvPoolReadout(
                    c_in,
                    c_out,
                    pool_ratio=self.pool_ratio,
                    sample=self.sample,
                    sparse=self.sparse_attn,
                    sl=self.sl,
                    lamb=self.lamb,
                    pool=use_pool,
                )
            )
        self.convpool_layers = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.lin3 = torch.nn.Linear(self.hidden_dim // 2, self.out_dim)

    def reset_parameters(self):
        for convpool in self.convpool_layers:
            convpool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, graph, n_feat):
        final_readout = None
        e_feat = None

        for i in range(self.num_layers):
            graph, n_feat, e_feat, readout = self.convpool_layers[i](
                graph, n_feat, e_feat
            )
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout

        n_feat = F.relu(self.lin1(final_readout))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = F.relu(self.lin2(n_feat))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = self.lin3(n_feat)

        return F.log_softmax(n_feat, dim=-1)

    def train_HGPSL(self, train_loader, optimizer):
        self.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        for batch in train_loader:
            optimizer.zero_grad()
            batch_graphs, batch_labels = batch
            batch_graphs = batch_graphs.to(self.device)
            batch_labels = batch_labels.reshape(-1).long().to(self.device)
            out = self(batch_graphs, batch_graphs.ndata["feat"])
            loss = F.nll_loss(out, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches
    
    def test(self, loader):
        self.eval()
        correct = 0.0
        loss = 0.0
        num_graphs = 0
        with torch.no_grad():
            for batch in loader:
                batch_graphs, batch_labels = batch
                num_graphs += batch_labels.size(0)
                batch_graphs = batch_graphs.to(self.device)
                batch_labels = batch_labels.reshape(-1).long().to(self.device)
                out = self(batch_graphs, batch_graphs.ndata["feat"])
                pred = out.argmax(dim=1)
                loss += F.nll_loss(out, batch_labels, reduction="sum").item()
                correct += pred.eq(batch_labels).sum().item()
        return correct / num_graphs, loss / num_graphs

    def fit(self, dataset):
        folds_results = []
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, self.folds))):

            train_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(train_idx), batch_size=self.batch_size)
            val_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(val_idx), batch_size=self.batch_size)
            test_loader = GraphDataLoader(dataset, sampler=SubsetRandomSampler(test_idx), batch_size=self.batch_size)

            self.reset_parameters()

            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

            # best_test_result, best_val = 0, float("inf")
            best_test_result, best_val = 0, float("-inf")
            for epoch in range(self.epochs):
                loss = self.train_HGPSL(train_loader, optimizer)
                train_result, _ = self.test(train_loader)
                val_result, val_loss = self.test(val_loader)
                test_result, _ = self.test(test_loader)

                print(f'Epoch: {epoch: 02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_result:.2f}%, '
                      f'Valid: {100 * val_result:.2f}%, '
                      f'Test: {100 * test_result:.2f}%')

                # if best_val > val_loss:
                #     best_val = val_loss
                if best_val < val_result:
                    print("Update Best Result!")
                    best_val = val_result
                    best_test_result = test_result
                    bad_cound = 0
                else:
                    bad_cound += 1

                if bad_cound >= self.patience:
                    break
            folds_results.append(best_test_result)
        
            print('Best result of fold {}: {}'.format(fold, best_test_result))
        
        print('\nTest results: ', folds_results)
        print('Final Test Accuracy: {:.4f}+{:.4f}'.format(np.mean(folds_results), np.std(folds_results)))


class WeightedGraphConv(GraphConv):
    r"""
    Description
    -----------
    GraphConv with edge weights on homogeneous graphs.
    If edge weights are not given, directly call GraphConv instead.

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform this operation.
    n_feat : torch.Tensor
        The node features
    e_feat : torch.Tensor, optional
        The edge features. Default: :obj:`None`
    """

    def forward(self, graph: DGLGraph, n_feat, e_feat=None):
        if e_feat is None:
            return super(WeightedGraphConv, self).forward(graph, n_feat)

        with graph.local_scope():
            if self.weight is not None:
                n_feat = torch.matmul(n_feat, self.weight)
            src_norm = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5)
            src_norm = src_norm.view(-1, 1)
            dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            dst_norm = dst_norm.view(-1, 1)
            n_feat = n_feat * src_norm
            graph.ndata["h"] = n_feat
            graph.edata["e"] = e_feat
            graph.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))
            n_feat = graph.ndata.pop("h")
            n_feat = n_feat * dst_norm
            if self.bias is not None:
                n_feat = n_feat + self.bias
            if self._activation is not None:
                n_feat = self._activation(n_feat)
            return n_feat


class ConvPoolReadout(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""

    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        pool_ratio=0.8,
        sample: bool = False,
        sparse: bool = True,
        sl: bool = True,
        lamb: float = 1.0,
        pool: bool = True,
        conv_type: str = 'GCN_Conv'
    ):
        super(ConvPoolReadout, self).__init__()
        self.use_pool = pool
        self.conv = WeightedGraphConv(in_feat, out_feat)
        if pool:
            self.pool = HGPSLPool(
                out_feat,
                ratio=pool_ratio,
                sparse=sparse,
                sample=sample,
                sl=sl,
                lamb=lamb,
            )
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.pool:
            self.pool.reset_parameters()

    def forward(self, graph, feature, e_feat=None):
        out = F.relu(self.conv(graph, feature, e_feat))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)
        readout = torch.cat(
            [self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1
        )
        return graph, out, e_feat, readout
    

class NodeInfoScoreLayer(nn.Module):
    r"""
    Description
    -----------
    Compute a score for each node for sort-pooling. The score of each node
    is computed via the absolute difference of its first-order random walk
    result and its features.

    Arguments
    ---------
    sym_norm : bool, optional
        If true, use symmetric norm for adjacency.
        Default: :obj:`True`

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform this operation.
    feat : torch.Tensor
        The node features
    e_feat : torch.Tensor, optional
        The edge features. Default: :obj:`None`

    Returns
    -------
    Tensor
        Score for each node.
    """

    def __init__(self, sym_norm: bool = True):
        super(NodeInfoScoreLayer, self).__init__()
        self.sym_norm = sym_norm

    def forward(self, graph: dgl.DGLGraph, feat: Tensor, e_feat: Tensor):
        with graph.local_scope():
            if self.sym_norm:
                src_norm = torch.pow(
                    graph.out_degrees().float().clamp(min=1), -0.5
                )
                src_norm = src_norm.view(-1, 1).to(feat.device)
                dst_norm = torch.pow(
                    graph.in_degrees().float().clamp(min=1), -0.5
                )
                dst_norm = dst_norm.view(-1, 1).to(feat.device)

                src_feat = feat * src_norm

                graph.ndata["h"] = src_feat
                graph.edata["e"] = e_feat
                graph = dgl.remove_self_loop(graph)
                graph.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))

                dst_feat = graph.ndata.pop("h") * dst_norm
                feat = feat - dst_feat
            else:
                dst_norm = 1.0 / graph.in_degrees().float().clamp(min=1)
                dst_norm = dst_norm.view(-1, 1)

                graph.ndata["h"] = feat
                graph.edata["e"] = e_feat
                graph = dgl.remove_self_loop(graph)
                graph.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))

                feat = feat - dst_norm * graph.ndata.pop("h")

            score = torch.sum(torch.abs(feat), dim=1)
            return score

edge_sparsemax = None
class HGPSLPool(nn.Module):
    r"""

    Description
    -----------
    The HGP-SL pooling layer from
    `Hierarchical Graph Pooling with Structure Learning <https://arxiv.org/pdf/1911.05954.pdf>`

    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels
    ratio : float, optional
        Pooling ratio. Default: 0.8
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency.
        Currently we only support full graph. Default: :obj:`False`
    sym_score_norm : bool, optional
        Use symmetric norm for adjacency or not. Default: :obj:`True`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    negative_slop : float, optional
        Negative slop for leaky_relu. Default: 0.2

    Returns
    -------
    DGLGraph
        The pooled graph.
    torch.Tensor
        Node features
    torch.Tensor
        Edge features
    torch.Tensor
        Permutation index
    """

    def __init__(
        self,
        in_feat: int,
        ratio=0.8,
        sample=True,
        sym_score_norm=True,
        sparse=True,
        sl=True,
        lamb=1.0,
        negative_slop=0.2,
        k_hop=3,
    ):
        super(HGPSLPool, self).__init__()
        self.in_feat = in_feat
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb
        self.negative_slop = negative_slop
        self.k_hop = k_hop

        self.att = Parameter(torch.Tensor(1, self.in_feat * 2))
        self.calc_info_score = NodeInfoScoreLayer(sym_norm=sym_score_norm)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.att.data)

    def forward(self, graph: DGLGraph, feat: Tensor, e_feat=None):
        # top-k pool first
        if e_feat is None:
            e_feat = torch.ones(
                (graph.num_edges(),), dtype=feat.dtype, device=feat.device
            )
        batch_num_nodes = graph.batch_num_nodes()
        x_score = self.calc_info_score(graph, feat, e_feat)
        perm, next_batch_num_nodes = topk(
            x_score, self.ratio, get_batch_id(batch_num_nodes), batch_num_nodes
        )
        feat = feat[perm]
        pool_graph = None
        if not self.sample or not self.sl:
            # pool graph
            graph.edata["e"] = e_feat
            pool_graph = dgl.node_subgraph(graph, perm)
            e_feat = pool_graph.edata.pop("e")
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        # no structure learning layer, directly return.
        if not self.sl:
            return pool_graph, feat, e_feat, perm

        # Structure Learning
        if self.sample:
            # A fast mode for large graphs.
            # In large graphs, learning the possible edge weights between each
            # pair of nodes is time consuming. To accelerate this process,
            # we sample it's K-Hop neighbors for each node and then learn the
            # edge weights between them.

            # first build multi-hop graph
            row, col = graph.all_edges()
            num_nodes = graph.num_nodes()

            scipy_adj = scipy.sparse.coo_matrix(
                (
                    e_feat.detach().cpu(),
                    (row.detach().cpu(), col.detach().cpu()),
                ),
                shape=(num_nodes, num_nodes),
            )
            for _ in range(self.k_hop):
                two_hop = scipy_adj**2
                two_hop = two_hop * (1e-5 / two_hop.max())
                scipy_adj = two_hop + scipy_adj
            row, col = scipy_adj.nonzero()
            row = torch.tensor(row, dtype=torch.long, device=graph.device)
            col = torch.tensor(col, dtype=torch.long, device=graph.device)
            e_feat = torch.tensor(
                scipy_adj.data, dtype=torch.float, device=feat.device
            )

            # perform pooling on multi-hop graph
            mask = perm.new_full((num_nodes,), -1)
            i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
            mask[perm] = i
            row, col = mask[row], mask[col]
            mask = (row >= 0) & (col >= 0)
            row, col = row[mask], col[mask]
            e_feat = e_feat[mask]

            # add remaining self loops
            mask = row != col
            num_nodes = perm.size(0)  # num nodes after pool
            loop_index = torch.arange(
                0, num_nodes, dtype=row.dtype, device=row.device
            )
            inv_mask = ~mask
            loop_weight = torch.full(
                (num_nodes,), 0, dtype=e_feat.dtype, device=e_feat.device
            )
            remaining_e_feat = e_feat[inv_mask]
            if remaining_e_feat.numel() > 0:
                loop_weight[row[inv_mask]] = remaining_e_feat
            e_feat = torch.cat([e_feat[mask], loop_weight], dim=0)
            row, col = row[mask], col[mask]
            row = torch.cat([row, loop_index], dim=0)
            col = torch.cat([col, loop_index], dim=0)

            # attention scores
            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(
                dim=-1
            )
            weights = (
                F.leaky_relu(weights, self.negative_slop) + e_feat * self.lamb
            )

            # sl and normalization
            sl_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(sl_graph, weights)
            else:
                weights = edge_softmax(sl_graph, weights)

            # get final graph
            mask = torch.abs(weights) > 0
            row, col, weights = row[mask], col[mask], weights[mask]
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)
            e_feat = weights

        else:
            # Learning the possible edge weights between each pair of
            # nodes in the pooled subgraph, relative slower.

            # construct complete graphs for all graph in the batch
            # use dense to build, then transform to sparse.
            # maybe there's more efficient way?
            batch_num_nodes = next_batch_num_nodes
            block_begin_idx = torch.cat(
                [
                    batch_num_nodes.new_zeros(1),
                    batch_num_nodes.cumsum(dim=0)[:-1],
                ],
                dim=0,
            )
            block_end_idx = batch_num_nodes.cumsum(dim=0)
            dense_adj = torch.zeros(
                (pool_graph.num_nodes(), pool_graph.num_nodes()),
                dtype=torch.float,
                device=feat.device,
            )
            for idx_b, idx_e in zip(block_begin_idx, block_end_idx):
                dense_adj[idx_b:idx_e, idx_b:idx_e] = 1.0
            row, col = torch.nonzero(dense_adj).t().contiguous()

            # compute weights for node-pairs
            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(
                dim=-1
            )
            weights = F.leaky_relu(weights, self.negative_slop)
            dense_adj[row, col] = weights

            # add pooled graph structure to weight matrix
            pool_row, pool_col = pool_graph.all_edges()
            dense_adj[pool_row, pool_col] += self.lamb * e_feat
            weights = dense_adj[row, col]
            del dense_adj
            torch.cuda.empty_cache()

            # edge softmax/sparsemax
            complete_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(complete_graph, weights)
            else:
                weights = edge_softmax(complete_graph, weights)

            # get new e_feat and graph structure, clean up.
            mask = torch.abs(weights) > 1e-9
            row, col, weights = row[mask], col[mask], weights[mask]
            e_feat = weights
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        return pool_graph, feat, e_feat, perm
    

def topk(
    x: torch.Tensor,
    ratio: float,
    batch_id: torch.Tensor,
    num_nodes: torch.Tensor,
):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.

    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.

    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0
    )

    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full(
        (batch_size * max_num_nodes,), torch.finfo(x.dtype).min
    )
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device)
        + i * max_num_nodes
        for i in range(batch_size)
    ]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k


def get_batch_id(num_nodes: torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.

    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full(
            (num_nodes[i],), i, dtype=torch.long, device=num_nodes.device
        )
        batch_ids.append(item)
    return torch.cat(batch_ids)


def _neighbor_sort(
    scores: Tensor,
    end_n_ids: Tensor,
    in_degrees: Tensor,
    cum_in_degrees: Tensor,
):
    """Sort edge scores for each node"""
    num_nodes, max_in_degree = in_degrees.size(0), int(in_degrees.max().item())

    # Compute the index for dense score matrix with size (N x D_{max})
    # Note that the end_n_ids here is the end_node tensor in dgl graph,
    # which is not grouped by its node id (i.e. in this form: 0,0,1,1,1,...,N,N).
    # Thus here we first sort the end_node tensor to make it easier to compute
    # indexs in dense edge score matrix. Since we will need the original order
    # for following gspmm and gsddmm operations, we also keep the reverse mapping
    # (the reverse_perm) here.
    end_n_ids, perm = torch.sort(end_n_ids)
    scores = scores[perm]
    _, reverse_perm = torch.sort(perm)

    index = torch.arange(
        end_n_ids.size(0), dtype=torch.long, device=scores.device
    )
    index = (index - cum_in_degrees[end_n_ids]) + (end_n_ids * max_in_degree)
    index = index.long()

    dense_scores = scores.new_full(
        (num_nodes * max_in_degree,), torch.finfo(scores.dtype).min
    )
    dense_scores[index] = scores
    dense_scores = dense_scores.view(num_nodes, max_in_degree)

    sorted_dense_scores, dense_reverse_perm = dense_scores.sort(
        dim=-1, descending=True
    )
    _, dense_reverse_perm = torch.sort(dense_reverse_perm, dim=-1)
    dense_reverse_perm = dense_reverse_perm + cum_in_degrees.view(-1, 1)
    dense_reverse_perm = dense_reverse_perm.view(-1)
    cumsum_sorted_dense_scores = sorted_dense_scores.cumsum(dim=-1).view(-1)
    sorted_dense_scores = sorted_dense_scores.view(-1)
    arange_vec = torch.arange(
        1, max_in_degree + 1, dtype=torch.long, device=end_n_ids.device
    )
    arange_vec = torch.repeat_interleave(
        arange_vec.view(1, -1), num_nodes, dim=0
    ).view(-1)

    valid_mask = sorted_dense_scores != torch.finfo(scores.dtype).min
    sorted_scores = sorted_dense_scores[valid_mask]
    cumsum_sorted_scores = cumsum_sorted_dense_scores[valid_mask]
    arange_vec = arange_vec[valid_mask]
    dense_reverse_perm = dense_reverse_perm[valid_mask].long()

    return (
        sorted_scores,
        cumsum_sorted_scores,
        arange_vec,
        reverse_perm,
        dense_reverse_perm,
    )


def _threshold_and_support_graph(
    gidx: HeteroGraphIndex, scores: Tensor, end_n_ids: Tensor
):
    """Find the threshold for each node and its edges"""
    in_degrees = _gspmm(gidx, "copy_rhs", "sum", None, torch.ones_like(scores))[
        0
    ]
    cum_in_degrees = torch.cat(
        [in_degrees.new_zeros(1), in_degrees.cumsum(dim=0)[:-1]], dim=0
    )

    # perform sort on edges for each node
    (
        sorted_scores,
        cumsum_scores,
        rhos,
        reverse_perm,
        dense_reverse_perm,
    ) = _neighbor_sort(scores, end_n_ids, in_degrees, cum_in_degrees)
    cumsum_scores = cumsum_scores - 1.0
    support = rhos * sorted_scores > cumsum_scores
    support = support[dense_reverse_perm]  # from sorted order to unsorted order
    support = support[reverse_perm]  # from src-dst order to eid order

    support_size = _gspmm(gidx, "copy_rhs", "sum", None, support.float())[0]
    support_size = support_size.long()
    idx = support_size + cum_in_degrees - 1

    # mask invalid index, for example, if batch is not start from 0 or not continuous, it may result in negative index
    mask = idx < 0
    idx[mask] = 0
    tau = cumsum_scores.gather(0, idx.long())
    tau /= support_size.to(scores.dtype)

    return tau, support_size


class EdgeSparsemaxFunction(Function):
    r"""
    Description
    -----------
    Pytorch Auto-Grad Function for edge sparsemax.

    We define this auto-grad function here since
    sparsemax involves sort and select, which are
    not derivative.
    """

    @staticmethod
    def forward(
        ctx,
        gidx: HeteroGraphIndex,
        scores: Tensor,
        eids: Tensor,
        end_n_ids: Tensor,
        norm_by: str,
    ):
        if not is_all(eids):
            gidx = gidx.edge_subgraph([eids], True).graph
        if norm_by == "src":
            gidx = gidx.reverse()

        # use feat - max(feat) for numerical stability.
        scores = scores.float()
        scores_max = _gspmm(gidx, "copy_rhs", "max", None, scores)[0]
        scores = _gsddmm(gidx, "sub", scores, scores_max, "e", "v")

        # find threshold for each node and perform ReLU(u-t(u)) operation.
        tau, supp_size = _threshold_and_support_graph(gidx, scores, end_n_ids)
        out = torch.clamp(_gsddmm(gidx, "sub", scores, tau, "e", "v"), min=0)
        ctx.backward_cache = gidx
        ctx.save_for_backward(supp_size, out)
        torch.cuda.empty_cache()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gidx = ctx.backward_cache
        supp_size, out = ctx.saved_tensors
        grad_in = grad_out.clone()

        # grad for ReLU
        grad_in[out == 0] = 0

        # dL/dv_i = dL/do_i - 1/k \sum_{j=1}^k dL/do_j
        v_hat = _gspmm(gidx, "copy_rhs", "sum", None, grad_in)[
            0
        ] / supp_size.to(out.dtype)
        grad_in_modify = _gsddmm(gidx, "sub", grad_in, v_hat, "e", "v")
        grad_in = torch.where(out != 0, grad_in_modify, grad_in)
        del gidx
        torch.cuda.empty_cache()

        return None, grad_in, None, None, None


def edge_sparsemax(graph: dgl.DGLGraph, logits, eids=ALL, norm_by="dst"):
    r"""
    Description
    -----------
    Compute edge sparsemax. For a node :math:`i`, edge sparsemax is an operation that computes

    .. math::
      a_{ij} = \text{ReLU}(z_{ij} - \tau(\z_{i,:}))

    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of sparsemax. :math:`\tau` is a function
    that can be found at the `From Softmax to Sparsemax <https://arxiv.org/pdf/1602.02068.pdf>`
    paper.

    NOTE: currently only homogeneous graphs are supported.

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform edge sparsemax on.
    logits : torch.Tensor
        The input edge feature.
    eids : torch.Tensor or ALL, optional
        A tensor of edge index on which to apply edge sparsemax. If ALL, apply edge
        sparsemax on all edges in the graph. Default: ALL.
    norm_by : str, could be 'src' or 'dst'
        Normalized by source nodes of destination nodes. Default: `dst`.

    Returns
    -------
    Tensor
        Sparsemax value.
    """
    # we get edge index tensors here since it is
    # hard to get edge index with HeteroGraphIndex
    # object without other information like edge_type.
    row, col = graph.all_edges(order="eid")
    assert norm_by in ["dst", "src"]
    end_n_ids = col if norm_by == "dst" else row
    if not is_all(eids):
        eids = astype(eids, graph.idtype)
        end_n_ids = end_n_ids[eids]
    return EdgeSparsemaxFunction.apply(
        graph._graph, logits, eids, end_n_ids, norm_by
    )


class EdgeSparsemax(torch.nn.Module):
    r"""
    Description
    -----------
    Compute edge sparsemax. For a node :math:`i`, edge sparsemax is an operation that computes

    .. math::
      a_{ij} = \text{ReLU}(z_{ij} - \tau(\z_{i,:}))

    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of sparsemax. :math:`\tau` is a function
    that can be found at the `From Softmax to Sparsemax <https://arxiv.org/pdf/1602.02068.pdf>`
    paper.

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform edge sparsemax on.
    logits : torch.Tensor
        The input edge feature.
    eids : torch.Tensor or ALL, optional
        A tensor of edge index on which to apply edge sparsemax. If ALL, apply edge
        sparsemax on all edges in the graph. Default: ALL.
    norm_by : str, could be 'src' or 'dst'
        Normalized by source nodes of destination nodes. Default: `dst`.

    NOTE: currently only homogeneous graphs are supported.

    Returns
    -------
    Tensor
        Sparsemax value.
    """

    def __init__(self):
        super(EdgeSparsemax, self).__init__()

    def forward(self, graph, logits, eids=ALL, norm_by="dst"):
        return edge_sparsemax(graph, logits, eids, norm_by)