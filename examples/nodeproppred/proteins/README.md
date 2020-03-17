# ogbn-proteins

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/mlp.py)**: Full-batch MLP training based on Node2Vec features. Requires Node2Vec node embeddings to be saved in `embedding.pt`. To generate them, please run `python node2vec.py`.
* **[Full-batch GCN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/full_batch.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`).
* **[Cluster-GIN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/cluster_gin.py)**: Mini-batch GNN training using a edge feature-based variant of the GIN model. To train with node features, please use the `--use_node_features` argument. Cluster-GNN examples require PyTorch Geometric > 1.4.2.

## Training & Evaluation

```
# Run with default config
python cluster_gin.py

# Run with custom config
python cluster_gin.py --hidden_channels=64
```
