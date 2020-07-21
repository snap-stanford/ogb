# ogbn-mag

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/mlp.py)**: Full-batch MLP training based on paper features and optional MetaPath2Vec features (`--use_node_embedding`). For training with MetaPath2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python metapath.py` [requires `torch-geometric>=1.5.0`].
* **[GNN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/gnn.py)**: Full-batch GNN training on the paper-paper relational graph using either the GCN or GraphSAGE operator (`--use_sage`) [requires `torch_geometric>=1.6.0`].
* **[R-GCN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/rgcn.py)**: Full-batch R-GCN training on the complete heterogeneous graph. This script will consume about 14GB of GPU memory [requires `torch_geometric>=1.4.3`].
* **[Cluster-GCN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/cluster_gcn.py)**: Mini-batch R-GCN training using the Cluster-GCN algorithm [requires `torch-geometric>= 1.4.3`].
* **[NeighborSampler](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/sampler.py)**: Mini-batch R-GCN training using neighbor sampling [requires `torch-geometric>=1.5.0`].
* **[GraphSAINT](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/graph_saint.py)**: Mini-batch R-GCN training using the GraphSAINT algorithm [requires `torch-geometric>=1.5.0`].

For the R-GCN implementation, we use distinct trainable node embeddings for all node types except for paper nodes.

## Training & Evaluation

```
# Run with default config
python graph_saint.py
```

