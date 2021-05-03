# ogbl-citation2

**Note (Dec 29, 2020)**: The older version `ogbl-citation` is deprecated because negative samples used in validation and test sets are found to be quite biased (i.e., half of the entity nodes are never sampled as negative examples). `ogbl-citation2` (available from `ogb>=1.2.4` ) fixes this issue while retaining everyelse the same. The leaderboard results of `ogbl-citation` and `ogbl-citation2` are *not* comparable. 

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/citation2/mlp.py)**: Full-batch MLP training based on paper features and optional Node2Vec features (`--use_node_embedding`). For training with Node2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py` [requires `torch-geometric>= 1.5.0`].
* **[GNN](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/citation2/gnn.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`). This script will require large amounts of GPU memory [requires `torch-geometric>=1.6.0`].
* **[Matrix Factorization](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/citation2/mf.py)**: Full-batch Matrix Factorization training.
* **[NeighborSampler](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/citation2/sampler.py)**: Mini-batch GNN training using neighbor sampling [requires `torch-geometric>=1.5.0`].
* **[Cluster-GCN](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/citation2/cluster_gcn.py)**: Mini-batch GCN training using the Cluster-GCN algorithm [requires `torch-geometric>= 1.4.3`].
* **[GraphSAINT](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/citation2/graph_saint.py)**: Mini-batch GCN training using the GraphSAINT algorithm [requires `torch-geometric>=1.5.0`].

## Training & Evaluation

```
# Run with default config
python cluster_gcn.py

# Run with custom config
python cluster_gcn.py --hidden_channels=128
```
