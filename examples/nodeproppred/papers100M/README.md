# ogbn-papers100M

This repository includes the following example scripts.

* **[sgc.py](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/papers100M/sgc.py)**: Simplified Graph Convolution (SGC) pre-processing on CPU. This will produce `sgc_dict.pt` that saves SGC features and labels for a subset of nodes that are associated with target labels. Requires more than 100GB CPU memory. Takes about a few hours to complete.
* **[node2vec.py](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/papers100M/node2vec.py)**: Node2vec training on CPU. This will produce `data_dict.pt` that saves features and labels for a subset of nodes that are associated with target labels. Requires more than 100GB CPU memory. Each epoch takes about a few weeks. The pre-trained output node2vec embedding (only for labeled nodes) is available [here](https://snap.stanford.edu/ogb/data/misc/ogbn_papers100M/data_dict.pt).
* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/papers100M/mlp.py)**: Mini-batch MLP training on GPU based on paper features and optional Node2Vec features (`--use_node_embedding`) or SGC features (`--use_sgc_embedding`).

Note that the graph is huge: the size of the downloaded file is 57GB, and the size of the Pytorch Geometric graph object is 79GB (which takes a while to be generated and saved).
Please ensure you have sufficient memory and storage before running the scripts.


## Training & Evaluation

```
# Generate SGC features and save them as sgc_dict.pt
python sgc.py

# Generate node2vec features and save them as data_dict.pt
# The features are saved every 1000 iterations.
python node2vec.py

### Train MLPs based on the saved features and labels
# Train based only on input node features (Requires data_dict.pt, the node2vec features do not need to be converged.)
python mlp.py

# Train based on SGC features (Requires sgc_dict.pt)
python mlp.py --use_sgc_embedding

# Train based on node2vec features (Requires data_dict.pt, the node2vec features need to be converged.)
python mlp.py --use_node_embedding
```

## Getting Raw Texts

The tsv file that maps paper IDs into their titles and abstracts are available [here](https://snap.stanford.edu/ogb/data/misc/ogbn_papers100M/paperinfo.zip) (34GB).
```bash
unzip paperinfo.zip
cd paperinfo
```

There are two files: `idx_title.tsv` and `idx_abs.tsv`. For `idx_title.tsv`, the format is nodeidx \t title. For `idx_abs.tsv`, the format is nodeidx \t abstract.
You can obtain the mapping from node idx to MAG's paper ID at `mapping/nodeidx2paperid.csv.gz` of the downloaded dataset directory.
Note that the titles and abstract were created from a MAG dump that is different from the original dataset; Hence, abstract and titles of some paper nodes are missing.
