<p align="center">
<img src="logo.jpeg" width="100%" class="center" alt="logo"/>
</p>

# Graph Structure Learning Benchmark (GSLB)

GSLB is a Graph Structure Learning (GSL) library and benchmark based on [DGL](https://github.com/dmlc/dgl) and [PyTorch](https://pytorch.org/). We integrate diverse datasets and state-of-the-art GSL models.

## 📔 What is Graph Structure Learning?

<p align="center">
<img src="pipeline.png" width="100%" class="center" alt="pipeline"/>
</p>

Graph Structure Learning (GSL) aims to optimize both the parameters of Graph Neural Networks (GNNs) and the computation graph structure simultaneously. GSL methods start with input features and an optimal initial graph structure. Its corresponding computation graph is iteratively refined through a structure learning module. With the refined computation graph ,GNNs are used to generate graph representations. Then parameters of the GNNs and the structure modeling module are jointly updated, either simultaneously or alternatively, util a preset stopping condition is satisfied.

If you want to explore more information about GSL, please refer to our [Paper](https://arxiv.org/abs/2310.05174), [survey](https://www.researchgate.net/profile/Yanqiao-Zhu/publication/349787551_Deep_Graph_Structure_Learning_for_Robust_Representations_A_Survey/links/6137188538818c2eaf885a3f/Deep-Graph-Structure-Learning-for-Robust-Representations-A-Survey.pdf), and [paper collection](https://github.com/GSL-Benchmark/Awesome-Graph-Structure-Learning).

## 🚀 Get Start

### Requirements

GSLB needs the following requirements to be satisfied beforehand:

* Python 3.8+
* PyTorch 1.13
* DGL 1.1+
* Scipy 1.9+
* Scikit-learn
* Numpy
* NetworkX
* ogb
* tqdm
* easydict
* PyYAML
* DeepRobust

### Installation via PyPI

To install GSLB with [`pip`](https://pip.pypa.io/en/stable/), simply run:

```
pip install GSLB
```

Then, you can import `GSL` from your current environment.

## Usage

If you want to quickly run an existing GSL model on a graph dataset:

```python
python main.py --dataset dataset_name --model model_name --num_trails --gpu_num 0 --use_knn --k 5 --use_mettack --sparse --metric acc --ptb_rate 0. --drop_rate 0. --add_rate 0.
```

Optional arguments:

``--dataset`` : the name of graph dataset

``--model`` : the name of GSL model

``--ntrail`` : repetition count of experiments

``--use_knn`` : whether to use knn graph instead of the original graph

``--k`` : the number of the nearest neighbors

``--drop_rate`` : the probability of randomly edge deletion

``--add_rate`` : the probability of randomly edge addition

``--mask_feat_rate`` : the probability of randomly mask features

``--use_mettack`` : whether to use the structure after being attacked by mettack

``--ptb_rate`` : the perturbation rate

``--metric`` : the evaluation metric

``--gpu_num`` : the selected GPU number

*Example: Train GRCN on Cora dataset, with the evaluation metric is accuracy.*

```
python main.py --dataset cora --model GRCN --metric acc
```

*If you want to quickly generate a perturbed graph by Mettack:*

```
python generate_attack.py --dataset cora --ptb_rate 0.05
```

Step 1: Load datasets

```python
from GSL.data import *

# load a homophilic or heterophilic graph dataset
data = Dataset(root='/tmp/', name='cora')

# load a perturbed graph dataset
data = Dataset(root='/tmp/', name='cora', use_mettack=True, ptb_rate=0.05)

# load a heterogeneous graph dataset
data = HeteroDataset(root='/tmp/', name='acm')

# load a graph-level dataset
data = GraphDataset(root='/tmp/', name='IMDB-BINARY', model='GCN')
```

Step 2: Initialize the GSL model

```python
from GSL.model import *
from GSL.utils import accuracy, macro_f1, micro_f1

model_name = 'GRCN'
metric = 'acc'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the hyper-parameters are recorded in config
config_path = './configs/{}_config.yaml'.format(model_name.lower())

# select a evaluation metric
eval_metric = {
  'acc': accuracy,
  'macro-f1': macro_f1,
  'micro-f1': micro_f1
}[metric]

model = GRCN(data.num_feat, data.num_class, evel_metric,
            config_path, dataset_name, device)
```

Step 3: Train GSL model

```python
model.fit(data)
```

## 🧩 Implementation Algorithms

Currently, we have implemented the following GSL algorithms:

| Algorithm | Conference/Journal | Category | Paper | Code |
| --------- | ------------------ | -------- | ----- | ---- |
| LDS | ICML 2019 | Homogeneous GSL | [Learning Discrete Structures for Graph Neural Networks](http://proceedings.mlr.press/v97/franceschi19a/franceschi19a.pdf) | [Link](https://github.com/lucfra/LDS-GNN) |
| GRCN | ECML PKDD 2020 | Homogeneous GSL | [Graph-Revised Convolutional Network](https://arxiv.org/pdf/1911.07123.pdf) | [Link](https://github.com/PlusRoss/GRCN) |
| ProGNN | KDD 202 | Homogeneous GSL | [Graph structure learning for robust graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049) | [Link](https://github.com/ChandlerBang/Pro-GNN) |
| IDGL | NeurIPS 2020 | Homogeneous GSL | [Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings](https://proceedings.neurips.cc/paper/2020/file/e05c7ba4e087beea9410929698dc41a6-Paper.pdf) | [Link](https://github.com/hugochan/IDGL) |
| GEN | WWW 2021 | Homogeneous GSL | [Graph Structure Estimation Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3442381.3449952?casa_token=Ac8pftvrgv0AAAAA:Ka_mklQVpQmYfhNVB-r66cf6fFsCdy8jyVKGFvzC1q5Ko5DbQQqci_3vopigN0jzTDlWiL8L8Q) | [Link](https://github.com/BUPT-GAMMA/Graph-Structure-Estimation-Neural-Networks) |
| CoGSL | WWW 2022 | Homogeneous GSL | [Compact Graph Structure Learning via Mutual Information Compression](https://dl.acm.org/doi/pdf/10.1145/3485447.3512206?casa_token=lyWPk8kyFzwAAAAA:HLmbgpzrKe17LbnQqNh2zI_6WOvNgm_VqfNAgEoqLSXR7rRm_Bzro1oNzETTQb63W9vcVlijNw) | [Link](https://github.com/liun-online/CoGSL) |
| SLAPS | NeurIPS 2021 | Homogeneous GSL | [SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks](https://proceedings.neurips.cc/paper/2021/file/bf499a12e998d178afd964adf64a60cb-Paper.pdf) | [Link](https://github.com/BorealisAI/SLAPS-GNN) |
| SUBLIME | WWW 2022 | Homogeneous GSL | [Towards Unsupervised Deep Graph Structure Learning](https://dl.acm.org/doi/pdf/10.1145/3485447.3512186?casa_token=445ECOqpVh4AAAAA:3ZBlGDSLFxrhwN0zEUdGFMpB4DslsI4h-rFLvI3cWHNzNsx6k-4m2t-NiDLubRvw1tBLaziISw) | [Link](https://github.com/GRAND-Lab/SUBLIME) |
| STABLE | KDD 2022 | Homogeneous GSL | [Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN](https://dl.acm.org/doi/abs/10.1145/3534678.3539484) | [Link](https://github.com/likuanppd/STABLE) |
| NodeFormer | NeurIPS 2022 | Homogeneous GSL | [NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classiﬁcation](https://openreview.net/forum?id=sMezXGG5So) | [Link](https://github.com/qitianwu/NodeFormer) |
| HES-GSL | TNNLS 2023 | Homogeneous GSL | [Homophily-Enhanced Self-Supervision for Graph Structure Learning: Insights and Directions](https://ieeexplore.ieee.org/abstract/document/10106110) | [Link](https://github.com/LirongWu/Homophily-Enhanced-Self-supervision) |
| GSR | WSDM 2023 | Homogeneous GSL | [Self-Supervised Graph Structure Refinement for Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3539597.3570455) | [Link](https://github.com/andyjzhao/WSDM23-GSR) |
| GTN | NeurIPS 2020 | Heterogeneous GSL | [Graph Transformer Networks](https://proceedings.neurips.cc/paper_files/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf) | [Link](https://github.com/seongjunyun/Graph_Transformer_Networks) |
| HGSL | AAAI 2021 | Heterogeneous GSL | [Heterogeneous Graph Structure Learning for Graph Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/16600) | [Link](https://github.com/AndyJZhao/HGSL) |
| HGP-SL | AAAI 2020 | Graph-level GSL | [Hierarchical Graph Pooling with Structure Learning](https://arxiv.org/abs/1911.05954) | [Link](https://github.com/cszhangzhen/HGP-SL) |
| VIB-GSL | AAAI 2022 | Graph-level GSL | [Graph Structure Learning with Variational Information Bottleneck](https://ojs.aaai.org/index.php/AAAI/article/view/20335) | [Link](https://github.com/RingBDStack/VIB-GSL) |


## Cite Us

Feel free to cite this work if you find it useful to you!
```
@article{li2023gslb,
  title={GSLB: The Graph Structure Learning Benchmark},
  author={Li, Zhixun and Wang, Liang and Sun, Xin and Luo, Yifan and Zhu, Yanqiao and Chen, Dingshuo and Luo, Yingtao and Zhou, Xiangxin and Liu, Qiang and Wu, Shu and others},
  journal={arXiv preprint arXiv:2310.05174},
  year={2023}
}
```
