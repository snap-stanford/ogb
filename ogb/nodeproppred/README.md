# Node Property Prediction
## 1. Datasets

### `ogbn-proteins`: Prediction of protein functions
This is an undirected, weighted, and typed graph (according to species), containing 132534 nodes. Nodes represent proteins, and edges indicate different types of biologically meaningful associations between proteins (e.g., physical interactions, co-expression, homology) [1,2]. 
The edges come with 8-dimensional features, where each dimension represents the strength of each association type and takes the value between 0 and 1 (the larger the value is, the stronger the association is).
The proteins come from 8 species:  E. coli, A. thaliana, S. cerevisiae, C. elegans, D. melanogaster, D. rerio, H. sapiens, M. musculus. The task is to predict the presence of protein functions (multi-label binary classification). There are 112 kinds of labels to predict in total.
We split the protein nodes into training/validation/test sets according to species from which the proteins come from.  

Note: The detail of this dataset is subject to change. Not yet fixed as a benchmark. 

### `ogbn-wiki`: Predict categories of Wikipedia articles
In preparation. Coming soon.

### `ogbn-products`: Prediction of product categories from Amazon co-purchasing network
In preparation. Coming soon. Adopt from [3].


## 2. Modules
### Data loader
We can obtain data loader for dataset named `d_name` as follows, where we can replace `d_name` with any available datasets (e.g., `"ogbn-proteins"`)

#### - Pytorch Geometric
```python
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset

dataset = PygNodePropPredDataset(name = d_name) 
num_tasks = dataset.num_tasks # obtaining number of prediction tasks in a dataset

splitted_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
graph = dataset[0] # pyg graph object
```

#### - DGL
```python
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset

dataset = DglNodePropPredDataset(name = d_name)
num_tasks = dataset.num_tasks # obtaining number of prediction tasks in a dataset

splitted_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
```
`{train,valid,test}_idx` are torch tensors of shape `(num_nodes,)`, representing the node indices assigned to training/validation/test sets.
Prediction target can be accessed by `graph.y`, which is a torch tensor of shape `(num_nodes, num_tasks)`, where the $i$-th row represents the target labels of $i$-th node.

### Evaluator
Evaluators are customized for each dataset.
We require users to pass pre-specified format to the evaluator.
First, please learn the input and output format specification of the evaluator as follows.

```python
from ogb.nodeproppred import Evaluator

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

[2] Ashburner, M., Ball, C.A., Blake, J.A., Botstein, D., Butler, H., Cherry, J.M., Davis, A.P., Dolinski, K., Dwight, S.S., Eppig, J.T. and Harris, M.A., 2000. Gene ontology: tool for the unification of biology. Nature Genetics, 25(1), p.25.

[3] Chiang, W. L., Liu, X., Si, S., Li, Y., Bengio, S., & Hsieh, C. J. Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks. SIGKDD 2019.
