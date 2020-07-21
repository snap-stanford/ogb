# ogbn-proteins

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/mlp.py)**: Full-batch MLP training based on average incoming edge features and optional Node2Vec features (`--use_node_embedding`). For training with Node2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py` [requires `torch-geometric>=1.5.0`].
* **[GNN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/gnn.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`) [requires `torch-geometric>=1.6.0`].

## Training & Evaluation

```
# Run with default config
python gnn.py

# Run with custom config (adding dropout may improve performance)
python gnn.py --dropout 0.5
```
