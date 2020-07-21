# ogbl-ddi

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ddi/mlp.py)**: Full-batch MLP training based on Node2Vec features. This script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py` [requires `torch-geometric>=1.5.0`].
* **[GNN](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ddi/gnn.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`) [requires `torch-geometric>=1.6.0`].
* **[Matrix Factorization](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ddi/mf.py)**: Full-batch Matrix Factorization training.

## Training & Evaluation

```
# Run with default config
python mlp.py

# Run with custom config
python mlp.py --hidden_channels=128
```
