# ogbn-products

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/mlp.py)**: Full-batch MLP training based on product features and optional Node2Vec features (`--use_node_embedding`). For training with Node2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py` [requires `torch-geometric>=1.5.0`].
* **[GNN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`). This script will consume large amounts of GPU memory [requires `torch_geometric>=1.6.0`].
* **[Cluster-GCN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/cluster_gcn.py)**: Mini-batch GCN training using the Cluster-GCN algorithm [requires `torch-geometric>= 1.4.3`].
* **[NeighborSampler](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py)**: Mini-batch GNN training using neighbor sampling [requires `torch-geometric>=1.5.0`].
* **[GraphSAINT](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/graph_saint.py)**: Mini-batch GCN training using the GraphSAINT algorithm [requires `torch-geometric>=1.5.0`].
* **[SIGN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/sign.py)**: Training based on pre-computed GNN representations using Scalable Inception Graph Neural Networks (SIGN) [requires `torch-geometric>=1.5.0`].

## Training & Evaluation

```
# Run with default config
python cluster_gcn.py

# Run with custom config
python cluster_gcn.py --hidden_channels=128
```
