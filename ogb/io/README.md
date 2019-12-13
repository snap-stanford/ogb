## Dataset format for node and link property prediction prediction
For the node and link prediction datasets, we download raw graph files and automatically preprocess them into either either Pytorch Geometric or DGL dataset object.

The following is a specification of downloaded graph files.
```
dataset/
├── processed
│   └── ...
├── raw
│   ├── edge-feat.csv.gz (optional)
│   ├── edge.csv.gz
│   └── node-feat.csv.gz (optional)
└── split
    └── split_name
        ├── test.csv
        ├── train.csv
        └── valid.csv
```

The `processed` directory provides the processed graph object, and is initially empty before preprocessing.

The `raw` directory provides the series of compressed csv files whose format is specified below. 

Let $|V|$ and $|E|$ be the number of nodes and edges, respectively. Let $K$ be the number of prediction tasks.
Nodes and edges are indexed from 0 to $|V| - 1$, and from 0 to $|E|-1$, respectively.

- **edge.csv.gz**: A matrix of shape $(|E|,2)$. Each row represents an edge represented by a pair of node indices. If the graph is undirected, this file only stores the single direction for each pair of nodes. The opposite edge will be automatically added when we create dataset object.
- **edge-feat.csv.gz**: A matrix of shape $(|E|, D_{e})$, where each row stores the $D_e$-dimensional edge feature. This file is optional to include.
- **node-feat.csv.gz**: A matrix of shape $(|V|, D_{v})$, where each row stores the $D_v$-dimensional node feature. This file is optional to include.

Finally, the `split/split_name` directory provides training, validation and test splits of nodes/edges in the graph, specified by a list of node indices. `split_name` is replaced with the name of appropriate splitting scheme for different datasets, e.g., `species` in the case of the `ogbn-proteins` dataset.

## Dataset format for graph property prediction
For graph property prediction datasets, we provide pre-processed Pytorch Geometric or DGL dataset object as well as raw graph files (e.g., containing list of SMILES strings for molecule datasets). Our module will directly utilize the preprocessed files to obtain dataset objects.

The example of downloadeded files are as follows in the case of Pytorch Geometrics.
```
tox21/
├── processed
│   └── geometric_data_processed.pt
├── raw
│   └── ...
└── split
    └── scaffold
        ├── test.csv.gz
        ├── train.csv.gz
        └── valid.csv.gz
```
In this example, our module will only touch `tox21/processed/geometric_data_processed.pt` and `tox21/split/scaffold/{train,valid,test}.csv.gz` for loading dataset and splitting dataset, respectively. The file(s) in the raw folder (`tox21/raw`) are just for the purpose of reference and will not be directly utilized by our module. 
