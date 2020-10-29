# ogbn-arxiv

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/mlp.py)**: Full-batch MLP training based on paper features and optional Node2Vec features (`--use_node_embedding`). For training with Node2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py` [requires `torch-geometric>=1.5.0`].
* **[GNN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`) [requires `torch-geometric>=1.6.0`].

## Training & Evaluation

```
# Run with default config
python gnn.py

# Run with custom config
python gnn.py --hidden_channels=128
```

## Getting Raw Texts

The tsv file that maps paper IDs into their titles and abstracts are available [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz).
There are three columns: paperid \t title \t abstract.
You can obtain the paper ID for each node at `mapping/nodeidx2paperid.csv.gz` of the downloaded dataset directory.
