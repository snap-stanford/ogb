# ogbl-collab

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/mlp.py)**: Full-batch MLP training based on author features and optional Node2Vec features (`--use_node_embedding`). For training with Node2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py` [requires `torch-geometric>=1.5.0`].
* **[GNN](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`) [requires `torch-geometric>=1.6.0`]. Setting `--use_valedges_as_input` would allow models to use validation edges at inference time. See [here](https://ogb.stanford.edu/docs/leader_rules/) for the rules of using validation labels.
* **[Matrix Factorization](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/mf.py)**: Full-batch Matrix Factorization training.

## Training & Evaluation

```
# Run with default config
python gnn.py

# Run with inference using validation edges
python gnn.py --use_valedges_as_input
```
