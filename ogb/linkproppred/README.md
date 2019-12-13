# Link Property Prediction
## 1. Datasets

### `ogbl-ppa`: Prediction of protein protein associations

`ogbl-ppa` is an undirected, unweighted graph, containing 576289 nodes. Nodes represent proteins and edges indicate biologically meaningful associations between proteins (e.g., physical interactions, co-expression, homology, genomic neighborhood) [1]. The task is to predict new association edges given already-existing ones. 
We provide a biological throughput split of the edges into training/validation/test edges, meaning that the goal is to predict a particular type of protein association (i.e., physical protein-protein interactions) from other types of protein associations (i.e., co-expression, homology, genomic neighborhood, etc.).

- Training edges: A list of edges that are present in the training graph. All the edges have positive labels (indicated by 1).
- Validation and test edges: A list of additional edges for evaluating link prediction models. We include both positive edges (unseen during training) and negative edges (randomly sampled).

The goal is to rank edges such that positive test edges score higher than negative test edges measured by ROC-AUC. We also provide a graph object constructed from training edges. 

Note: The detail of this dataset is subject to change. Not yet fixed as a benchmark. 


### `ogbl-ddi`: Prediction of drug drug interaction
In preparation. Coming soon.

### `ogbl-biomed`: Prediction of missing relations in human biomedical knowledge graph
In preparation. Coming soon.

### `ogbl-reviews`: Prediction of Amazon review ratings
In preparation. Coming soon.

### `ogbl-citations`: Prediction of paper citation link in academic graph
In preparation. Coming soon.


## 2. Modules
### Data loader
We can obtain data loader for dataset named `d_name` as follows, where we can replace `d_name` with any available datasets (e.g., `"ogbl-ppa"`)

#### - Pytorch Geometric
```python
from ogb.linkproppred.dataset_pyg import PygLinkPropPredDataset

dataset = PygLinkPropPredDataset(name = d_name) 
splitted_edge = dataset.get_edge_split()
train_edge, train_edge_label = splitted_edge["train_edge"], splitted_edge["train_edge_label"]
valid_edge, valid_edge_label = splitted_edge["valid_edge"], splitted_edge["valid_edge_label"]
test_edge, test_edge_label = splitted_edge["test_edge"], splitted_edge["test_edge_label"]
graph = dataset[0] # pyg graph object containing only training edges
```

#### - DGL
```python
from ogb.linkproppred.dataset_dgl import DglLinkPropPredDataset

dataset = DglLinkPropPredDataset(name = d_name)
splitted_edge = dataset.get_edge_split()

train_edge, train_edge_label = splitted_edge["train_edge"], splitted_edge["train_edge_label"]
valid_edge, valid_edge_label = splitted_edge["valid_edge"], splitted_edge["valid_edge_label"]
test_edge, test_edge_label = splitted_edge["test_edge"], splitted_edge["test_edge_label"]
graph = dataset[0] # dgl graph object containing only training edges
```
`{train,valid,edge}_edge` are torch tensors of size `(num_edge,2)`, where each row represents a directed edge, and the first/second column represents the source/sink node indices. 
An undirected graph should include bidirectional edges for each pair of nodes that are connected by an edge. We include the bidirectional edges in the graph object so that messages in GNNs flow in both directions. To keep a low-memory footprint, we did not duplicate edges in `{train,valid,edge}_edge`.
`{train,valid,edge}_edge_label` are torch tensors of length `num_edge`, where the specific shape depends on the dataset at hand. The $i$-th entry of `{train,valid,edge}_edge_label` corresponds to that of `{train,valid,edge}_edge`, representing some label(s) assigned to the $i$-th edge.

### Evaluator
Evaluators are customized for each dataset.
We require users to pass pre-specified format to the evaluator.
First, please learn the input and output format specification of the evaluator as follows.

```python
from ogb.linkproppred import Evaluator

evaluator = Evaluator(name = d_name)
print(evaluator.expected_input_format) 
print(evaluator.expected_output_format) 
```

Then, you can pass input dictionary (denoted by `input_dict` below) of specified format, and get the performance of your prediction.

```python
# In most cases, input_dict is
# input_dict = {"y_true": y_true, "y_pred": y_pred}
result_dict = evaluator.eval(input_dict)
```

## Reference
[1] Szklarczyk, D., Gable, A.L., Lyon, D., Junge, A., Wyder, S., Huerta-Cepas, J., Simonovic, M., Doncheva, N.T., Morris, J.H., Bork, P. and Jensen, L.J., 2018. STRING v11: protein–protein association networks with increased coverage, supporting functional discovery in genome-wide experimental datasets. Nucleic Acids Research, 47(D1), pp.D607-D613.

