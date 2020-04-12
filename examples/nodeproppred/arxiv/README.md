# ogbn-arxiv

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/mlp.py)**: Full-batch MLP training based on paper features and optional Node2Vec features (`--use_node_embedding`). For training with Node2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py`.
* **[Full-batch GCN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/full_batch.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`).

## Training & Evaluation

```
# Run with default config
python full_batch.py

# Run with custom config
python full_batch.py --hidden_channels=128
```
