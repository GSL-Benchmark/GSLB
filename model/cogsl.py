import time
import os.path as osp
from scipy.linalg import fractional_matrix_power, inv
from GSL.model import BaseModel
from GSL.utils import *
from GSL.learner import *
from GSL.encoder import *
from GSL.metric import *
from GSL.processor import *


class CoGSL(BaseModel):
    def __init__(self, num_features, num_classes, metric, config_path, dataset_name, device, data):
        super(CoGSL, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device)

        self.cogsl_data = construct_cogsl_data(data.name, data.adj, data.features, data.labels, data.train_mask, data.val_mask, data.test_mask, self.config.adj_hop).to(device)
        self.cls = Classification(num_features, self.config.cls_hid_1, num_classes, self.config.cls_dropout)
        self.ve = ViewEstimator(self.cogsl_data, num_features, self.config.gen_hid, self.config.com_lambda_v1, self.config.com_lambda_v2, self.config.ve_dropout)
        self.mi = MI_NCE(num_features, self.config.mi_hid_1, self.config.tau, self.config.big, self.config.batch)
        self.fusion = Fusion(self.config.lam, self.config.alpha, self.config.dataset)

    def get_view(self, data):
        new_v1, new_v2 = self.ve(data)
        return new_v1, new_v2

    def get_mi_loss(self, feat, views):
        mi_loss = self.mi(views, feat)
        return mi_loss

    def get_cls_loss(self, v1, v2, feat):
        prob_v1 = self.cls(feat, v1, "v1")
        prob_v2 = self.cls(feat, v2, "v2")
        logits_v1 = torch.log(prob_v1 + 1e-8)
        logits_v2 = torch.log(prob_v2 + 1e-8)
        return logits_v1, logits_v2, prob_v1, prob_v2

    def get_v_cls_loss(self, v, feat):
        logits = torch.log(self.cls(feat, v, "v") + 1e-8)
        return logits

    def get_fusion(self, v1, prob_v1, v2, prob_v2):
        v = self.fusion(v1, prob_v1, v2, prob_v2)
        return v

    def loss_acc(self, output, y):
        loss = F.nll_loss(output, y)
        acc = accuracy(output, y)
        return loss, acc

    def train_mi(self, x, views):
        vv1, vv2, v1v2 = self.get_mi_loss(x, views)
        return self.config.mi_coe * v1v2 + (vv1 + vv2) * (1 - self.config.mi_coe) / 2

    def train_cls(self):
        new_v1, new_v2 = self.get_view(self.data)
        logits_v1, logits_v2, prob_v1, prob_v2 = self.get_cls_loss(new_v1, new_v2, self.data.x)
        curr_v = self.get_fusion(new_v1, prob_v1, new_v2, prob_v2)
        logits_v = self.get_v_cls_loss(curr_v, self.data.x)

        views = [curr_v, new_v1, new_v2]

        loss_v1, _ = self.loss_acc(logits_v1[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_v2, _ = self.loss_acc(logits_v2[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_v, _ = self.loss_acc(logits_v[self.data.train_mask], self.data.y[self.data.train_mask])
        return self.config.cls_coe * loss_v + (loss_v1 + loss_v2) * (1 - self.config.cls_coe) / 2, views

    def fit(self, data):
        self.data = self.cogsl_data
        self.opti_ve = torch.optim.Adam(self.ve.parameters(), lr=self.config.ve_lr,
                                        weight_decay=self.config.ve_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opti_ve, 0.99)

        self.opti_cls = torch.optim.Adam(self.cls.parameters(), lr=self.config.cls_lr,
                                         weight_decay=self.config.cls_weight_decay)
        self.opti_mi = torch.optim.Adam(self.mi.parameters(), lr=self.config.mi_lr,
                                        weight_decay=self.config.mi_weight_decay)

        self.best_acc_val = 0
        self.best_loss_val = 1e9
        self.best_test = 0
        self.best_v = None
        self.own_str = self.config.dataset

        for epoch in range(self.config.main_epoch):
            curr = np.log(1 + self.config.temp_r * epoch)
            curr = min(max(0.05, curr), 0.1)

            for inner_ne in range(self.config.inner_ne_epoch):
                self.train()
                self.opti_ve.zero_grad()
                cls_loss, views = self.train_cls()
                mi_loss = self.train_mi(self.data.x, views)
                loss = cls_loss - curr * mi_loss
                loss.backward()
                self.opti_ve.step()
            self.scheduler.step()

            for inner_cls in range(self.config.inner_cls_epoch):
                self.train()
                self.opti_cls.zero_grad()
                cls_loss, _ = self.train_cls()
                cls_loss.backward()
                self.opti_cls.step()

            for inner_mi in range(self.config.inner_mi_epoch):
                self.train()
                self.opti_mi.zero_grad()
                _, views = self.train_cls()
                mi_loss = self.train_mi(self.data.x, views)
                mi_loss.backward()
                self.opti_mi.step()

            ## validation ##
            self.eval()
            _, views = self.train_cls()
            logits_v_val = self.get_v_cls_loss(views[0], self.data.x)
            loss_val, acc_val = self.loss_acc(logits_v_val[self.data.val_mask], self.data.y[self.data.val_mask])
            test_result = self.test(views[0])
            if acc_val >= self.best_acc_val and self.best_loss_val > loss_val:
                self.best_acc_val = max(acc_val, self.best_acc_val)
                self.best_loss_val = loss_val
                self.best_v = views[0]
                self.best_result = test_result.item()
            print(f'Epoch: {epoch: 02d}, '
                  f'Val Loss: {loss_val:.4f}, '
                  f'Valid: {100 * acc_val:.2f}%, '
                  f'Test: {100 * test_result:.2f}%')

    def test(self, adj):
        with torch.no_grad():
            self.eval()
            probs = self.cls.encoder_v(self.data.x, adj)
            accu = accuracy(probs[self.data.test_mask], self.data.y[self.data.test_mask])
            return accu


class CoGSLDataset():
    def __init__(self, dataset_name, x, y, view1, view2, view1_indices, view2_indices, train_mask, val_mask, test_mask):
        self.dataset_name = dataset_name
        self.x = x
        self.y = y
        self.view1 = view1
        self.view2 = view2
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_node = x.size(0)
        self.num_feature = x.size(1)
        self.num_class = int(torch.max(y)) + 1
        self.v1_indices = view1_indices
        self.v2_indices = view2_indices

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.view1 = self.view1.to(device)
        self.view2 = self.view2.to(device)
        self.v1_indices = self.v1_indices.to(device)
        self.v2_indices = self.v2_indices.to(device)
        return self

    def normalize(self, adj):
        if self.dataset_name in ["wikics", "ms"]:
            adj_ = (adj + adj.t())
            normalized_adj = adj_
        else:
            adj_ = (adj + adj.t())
            normalized_adj = self._normalize(adj_ + torch.eye(adj_.shape[0]).to(adj.device).to_sparse())
        return normalized_adj

    def _normalize(self, mx):
        mx = mx.to_dense()
        rowsum = mx.sum(1) + 1e-6  # avoid NaN
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx.to_sparse()


def construct_cogsl_data(dataset_name, adj, features, labels, train_mask, val_mask, test_mask, adj_hop=1):
    print("Constructing {} dataset for CoGSL...".format(dataset_name))
    features = row_normalize_features(features)

    # real-time computation of the graph structure for two views
    if adj[0, 0] == 0:
        adj += np.eye(adj.shape[0])
        print("self-loop!")

    def diff_knn(adj, alpha, k):
        def _normalize(mx):
            mx = mx.to_dense()
            rowsum = mx.sum(1) + 1e-6  # avoid NaN
            r_inv = rowsum.pow(-1 / 2).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)
            mx = r_mat_inv @ mx
            mx = mx @ r_mat_inv
            return mx.to_sparse()
        at = _normalize(adj)
        adj = alpha * torch.linalg.inv((torch.eye(adj.shape[0]) - (1 - alpha) * at))
        adj = sp.coo_matrix(adj)
        knn_adj = topk(k, adj)
        knn_adj = sp.coo_matrix(knn_adj)
        return knn_adj


    def get_khop_indices(k, view):
        view = (view.A > 0).astype("int32")
        view_ = view
        for i in range(1, k):
            view_ = (np.matmul(view_, view.T) > 0).astype("int32")
        view_ = torch.tensor(view_).to_sparse()
        return view_.indices()

    def topk(k, adj):
        adj = adj.toarray()
        pos = np.zeros(adj.shape)
        for i in range(adj.shape[0]):
            one = adj[i].nonzero()[0]
            if len(one) > k:
                oo = np.argsort(-adj[i, one])
                sele = one[oo[:k]]
                pos[i, sele] = adj[i, sele]
            else:
                pos[i, one] = adj[i, one]
        return pos

    # v1: adj, v2: diff
    print('Compute view1 adj...')
    start_time = time.time()
    ori_view1 = sp.coo_matrix(adj)
    end_time = time.time()
    print("Execution time: ", end_time - start_time, " seconds.")

    print('Compute view2 adj...')
    start_time = time.time()
    file_path = osp.join(osp.expanduser('~'), "datasets/" + dataset_name + "/" + "diff_40knn.npz")
    if os.path.isfile(file_path):
        ori_view2 = sp.load_npz(file_path)
    else:
        print('The first time may take some time, please be patient.')
        ori_view2 = diff_knn(adj, alpha=0.2, k=40)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        sp.save_npz(file_path, ori_view2)
    end_time = time.time()
    print("Execution time: ", end_time - start_time, " seconds.")

    print('Compute view1 indices...')
    start_time = time.time()
    file_path = osp.join(osp.expanduser('~'), "datasets/" + dataset_name + "/" + "adj_"+str(adj_hop)+"hop.pt")
    if os.path.isfile(file_path):
        ori_view1_indices = torch.load(file_path)
    else:
        print('The first time may take some time, please be patient.')
        ori_view1_indices = get_khop_indices(adj_hop, ori_view1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(ori_view1_indices, file_path)
    end_time = time.time()
    print("Execution time: ", end_time - start_time, " seconds.")

    print('Compute view2 indices...')
    start_time = time.time()
    file_path = osp.join(osp.expanduser('~'), "datasets/" + dataset_name + "/" + "diff_40knn_1hop.pt")
    if os.path.isfile(file_path):
        ori_view2_indices = torch.load(file_path)
    else:
        print('The first time may take some time, please be patient.')
        ori_view2_indices = get_khop_indices(1, ori_view2)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(ori_view2_indices, file_path)
    end_time = time.time()
    print("Execution time: ", end_time - start_time, " seconds.")

    ori_view1 = normalize(sparse_mx_to_torch_sparse_tensor(ori_view1), "sym", True)
    ori_view2 = normalize(sparse_mx_to_torch_sparse_tensor(ori_view2), "sym", True)

    return CoGSLDataset(dataset_name=dataset_name, x=features, y=labels, view1=ori_view1, view2=ori_view2,
                        view1_indices=ori_view1_indices, view2_indices=ori_view2_indices,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


class Classification(nn.Module):
    def __init__(self, num_feature, cls_hid_1, num_class, dropout):
        super(Classification, self).__init__()
        self.encoder_v1 = GCN(num_feature, cls_hid_1, num_class, 2, dropout, 0., False, conv_bias=True)
        self.encoder_v2 = GCN(num_feature, cls_hid_1, num_class, 2, dropout, 0., False, conv_bias=True)
        self.encoder_v = GCN(num_feature, cls_hid_1, num_class, 2, dropout, 0., False, conv_bias=True)

    def forward(self, feat, view, flag):
        if flag == "v1":
            prob = F.softmax(self.encoder_v1(feat, view), dim=1)
        elif flag == "v2":
            prob = F.softmax(self.encoder_v2(feat, view), dim=1)
        elif flag == "v":
            prob = F.softmax(self.encoder_v(feat, view), dim=1)
        return prob


class GenView(nn.Module):
    def __init__(self, num_feature, hid, com_lambda, dropout, adj):
        super(GenView, self).__init__()
        self.com_lambda = com_lambda

        metric = MLPRefineSimilarity(hid, dropout).to(adj.device)
        processors = [Normalize(mode='row_softmax_sparse')]
        self.graph_learner = GNNLearner(metric, processors, adj, 1, num_feature, hid, nn.ReLU(), False, 0, 'GCN', bias=True)

    def forward(self, v_ori, feat, v_indices):
        new_adj = self.graph_learner(feat, v_indices)
        new_adj = new_adj.to_sparse()
        gen_v = v_ori + self.com_lambda * new_adj
        return gen_v


class ViewEstimator(nn.Module):
    def __init__(self, data, num_feature, gen_hid, com_lambda_v1, com_lambda_v2, dropout):
        super(ViewEstimator, self).__init__()
        self.v1_gen = GenView(num_feature, gen_hid, com_lambda_v1, dropout, data.view1)
        self.v2_gen = GenView(num_feature, gen_hid, com_lambda_v2, dropout, data.view2)

    def forward(self, data):
        new_v1 = data.normalize(self.v1_gen(data.view1, data.x, data.v1_indices))
        new_v2 = data.normalize(self.v2_gen(data.view2, data.x, data.v2_indices))
        return new_v1, new_v2


class MI_NCE(nn.Module):
    def __init__(self, num_feature, mi_hid_1, tau, big, batch):
        super(MI_NCE, self).__init__()
        self.gcn = GCNConv(num_feature, mi_hid_1, bias=True, activation=nn.PReLU())
        self.gcn1 = GCNConv(num_feature, mi_hid_1, bias=True, activation=nn.PReLU())
        self.gcn2 = GCNConv(num_feature, mi_hid_1, bias=True, activation=nn.PReLU())

        self.proj = nn.Sequential(
            nn.Linear(mi_hid_1, mi_hid_1),
            nn.ELU(),
            nn.Linear(mi_hid_1, mi_hid_1)
        )
        self.con = Contrast(tau)
        self.big = big
        self.batch = batch

    def forward(self, views, feat):
        v_emb = self.proj(self.gcn(feat, views[0], sparse=True))
        v1_emb = self.proj(self.gcn1(feat, views[1], sparse=True))
        v2_emb = self.proj(self.gcn2(feat, views[2], sparse=True))
        # if dataset is so big, we will randomly sample part of nodes to perform MI estimation
        if self.big == True:
            idx = np.random.choice(feat.shape[0], self.batch, replace=False)
            idx.sort()
            v_emb = v_emb[idx]
            v1_emb = v1_emb[idx]
            v2_emb = v2_emb[idx]

        vv1 = self.con.cal(v_emb, v1_emb)
        vv2 = self.con.cal(v_emb, v2_emb)
        v1v2 = self.con.cal(v1_emb, v2_emb)

        return vv1, vv2, v1v2


class Contrast:
    def __init__(self, tau):
        self.tau = tau

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def cal(self, z1_proj, z2_proj):
        matrix_z1z2 = self.sim(z1_proj, z2_proj)
        matrix_z2z1 = matrix_z1z2.t()

        matrix_z1z2 = matrix_z1z2 / (torch.sum(matrix_z1z2, dim=1).view(-1, 1) + 1e-8)
        lori_v1v2 = -torch.log(matrix_z1z2.diag()+1e-8).mean()

        matrix_z2z1 = matrix_z2z1 / (torch.sum(matrix_z2z1, dim=1).view(-1, 1) + 1e-8)
        lori_v2v1 = -torch.log(matrix_z2z1.diag()+1e-8).mean()
        return (lori_v1v2 + lori_v2v1) / 2

class Fusion(nn.Module):
    def __init__(self, lam, alpha, name):
        super(Fusion, self).__init__()
        self.lam = lam
        self.alpha = alpha
        self.name = name

    def get_weight(self, prob):
        out, _ = prob.topk(2, dim=1, largest=True, sorted=True)
        fir = out[:, 0]
        sec = out[:, 1]
        w = torch.exp(self.alpha*(self.lam*torch.log(fir+1e-8) + (1-self.lam)*torch.log(fir-sec+1e-8)))
        return w

    def forward(self, v1, prob_v1, v2, prob_v2):
        w_v1 = self.get_weight(prob_v1)
        w_v2 = self.get_weight(prob_v2)
        beta_v1 = w_v1 / (w_v1 + w_v2)
        beta_v2 = w_v2 / (w_v1 + w_v2)
        if self.name not in ["citeseer", "digits", "polblogs"]:
            beta_v1 = beta_v1.diag().to_sparse()
            beta_v2 = beta_v2.diag().to_sparse()
            v = torch.sparse.mm(beta_v1, v1) + torch.sparse.mm(beta_v2, v2)
            return v
        else :
            beta_v1 = beta_v1.unsqueeze(1)
            beta_v2 = beta_v2.unsqueeze(1)
            v = beta_v1 * v1.to_dense() + beta_v2 * v2.to_dense()
            return v.to_sparse()
