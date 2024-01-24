import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_similarity_graph(node_embeddings, right_node_embedding=None):
    if right_node_embedding is None:
        similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    else:
        similarity_graph = torch.mm(node_embeddings, right_node_embedding.t())
    return similarity_graph


class InnerProductSimilarity:
    def __init__(self):
        super(InnerProductSimilarity, self).__init__()

    def __call__(self, embeddings, right_embeddings=None):
        similarities = cal_similarity_graph(embeddings, right_embeddings)
        return similarities


class CosineSimilarity:
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def __call__(self, embeddings, right_embeddings=None):
        if right_embeddings is None:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
        else:
            embeddings = F.normalize(embeddings, dim=1, p=2)
            right_embeddings = F.normalize(right_embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings, right_embeddings)
        return similarities


class WeightedCosine(nn.Module):
    def __init__(self, input_size, num_pers):
        super().__init__()
        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def __call__(self, embeddings):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        if len(embeddings.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

        embeddings_fc = embeddings.unsqueeze(0) * expand_weight_tensor
        embeddings_norm = F.normalize(embeddings_fc, p=2, dim=-1)
        attention = torch.matmul(
            embeddings_norm, embeddings_norm.transpose(-1, -2)
        ).mean(0)
        return attention


class MLPRefineSimilarity(nn.Module):
    def __init__(self, hid, dropout):
        super(MLPRefineSimilarity, self).__init__()
        self.gen_mlp = nn.Linear(2 * hid, 1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, embeddings, v_indices):
        num_node = embeddings.shape[0]
        f1 = embeddings[v_indices[0]]
        f2 = embeddings[v_indices[1]]
        ff = torch.cat([f1, f2], dim=-1)
        temp = self.gen_mlp(self.dropout(ff)).reshape(-1)
        z_matrix = torch.sparse.FloatTensor(v_indices, temp, (num_node, num_node)).to_dense()
        return z_matrix

# class Minkowski:
