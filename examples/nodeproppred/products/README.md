# ogbn-products

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/mlp.py)**: Full-batch MLP training based on product features and optional Node2Vec features (`--node_embedding`). For training with Node2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py`.
* **[Full-batch GCN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/full_batch.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`). This script will require large amounts of GPU memory.
* **[Cluster-GNN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/cluster_gcn.py)**: Mini-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`). Cluster-GNN examples require PyTorch Geometric > 1.4.2.

## Training & Evaluation

```
# Run with default config
python cluster_gcn.py

# Run with custom config
python cluster_gcn.py --hidden_channels=128
```
