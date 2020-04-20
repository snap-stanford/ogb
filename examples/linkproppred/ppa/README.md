# ogbl-ppa

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ppa/mlp.py)**: Full-batch MLP training based on species ID features and optional Node2Vec features (`--use_node_embedding`). For training with Node2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py`.
* **[Full-batch GCN](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ppa/full_batch.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`).
* **[Matrix Factorization](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ppa/mf.py)**: Full-batch Matrix Factorization training.

## Training & Evaluation

```
# Run with default config
python mlp.py

# Run with custom config
python mlp.py --hidden_channels=128
```
