# DatasetSaver

The `DatasetSaver` class allows external contributors to prepare their datasets in OGB-compatible manner.

Below is the quick example of how to use `DatasetSaver` class, where we focus on graph property prediction datasets.
Please follow the steps below **in the exact order** to generate final dataset files.

## 1. Constructor
Create a constructor of `DatasetSaver`. `dataset_name` needs to follow OGB convention and start from either `ogbn-`, `ogbl-`, or `ogbg-`. `is_hetero` is `True` for heterogeneous graphs, and `version` indicates the dataset version.
```python
from ogb.io import DatasetSaver
import numpy as np
import networkx as nx
import os

# constructor
dataset_name = 'ogbg-toy'
saver = DatasetSaver(dataset_name = dataset_name, is_hetero = False, version = 1)
```

## 2. Saving graph list

Create `graph_list`, storing your graph objects, and call `saver.save_graph_list(graph_list)`. 

Graph objects are dictionaries containing the following keys.
### Homogeneous graph:
- `edge_index` (necessary): `numpy.ndarray` of shape `(2, num_edges)`. Please include bidirectional edges explicitly if graphs are undirected.
- `num_nodes` (necessary): `int`, denoting the number of nodes in the graph.
- `node_feat` (optional): `numpy.ndarray` of shape `(num_nodes, node_feat_dim)`.
- `edge_feat` (optional): `numpy.ndarray` of shape `(num_edges, edge_feat_dim)`. 

### Heterogeneous graph:
- `edge_index_dict` (necessary): Dictionary mapping triplets `(head type, relation type, tail type)` to `edge_index`
- `num_nodes_dict` (necessary): Dictionary mapping `entity type` to `num_nodes`.
- `node_feat_dict` (optional): Dictionary mapping `entity type` to `node_feat`.
- `edge_feat_dict` (optional): Dictionary mapping `(head type, relation type, tail type)` to `edge_feat`.

```python
# generate random graphs with node and edge features
graph_list = []
num_data = 100
for i in range(num_data):
    g = nx.fast_gnp_random_graph(10, 0.5)
    graph = dict()
    graph['edge_index'] = np.array(g.edges).transpose() 
    num_edges = graph['edge_index'].shape[1]

    graph['num_nodes'] = len(g.nodes)
    # optionally, add node/edge features
    graph['node_feat'] = np.random.randn(graph['num_nodes'], 3)
    graph['edge_feat'] = np.random.randn(num_edges, 3) 
    
    graph_list.append(graph)

# saving a list of graphs
saver.save_graph_list(graph_list)
```

## 3. Saving target labels
Save target labels to predict. Only needed for graph/node property prediction datasets.
```python
num_classes = 3
labels = np.random.randint(num_classes, size = (num_data,1))
saver.save_target_labels(labels)
```

## 4. Saving dataset split
Prepare `split_idx`, a dictionary with three keys, `train`, `valid`, and `test`, and mapping into data indices of `numpy.ndarray`. Then, call `saver.save_split(split_idx, split_name = xxx)`.
```python
split_idx = dict()
perm = np.random.permutation(num_data)
split_idx['train'] = perm[:int(0.8*num_data)]
split_idx['valid'] = perm[int(0.8*num_data): int(0.9*num_data)]
split_idx['test'] = perm[int(0.9*num_data):]
saver.save_split(split_idx, split_name = 'random')
```

## 5. Copying mapping directory
Store all the mapping information and `README.md` in `mapping_path` and call `saver.copy_mapping_dir(mapping_path)`.

```python
mapping_path = 'mapping/'

# prepare mapping information first and store it under this directory (empty below).
os.makedirs(mapping_path)
os.mknod(os.path.join(mapping_path, 'README.md'))

saver.copy_mapping_dir(mapping_path)
```

## 6. Saving task information
Save task information by calling `saver.save_task_info(task_type, eval_metric, num_classes = num_classes)`.
`eval_metric` is used to call `Evaluator` (c.f. [here](https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py)). You can reuse one of the existing metrics, or you can implement your own by creating a pull request.
```python
saver.save_task_info(task_type = 'classification', eval_metric = 'acc', num_classes = num_classes)
```
 
## 7. Getting meta information dictionary
```python
meta_dict = saver.get_meta_dict()
```

## 8. Testing the dataset object
Test the OGB dataset object to confirm it is working as you expect. You can similarly test Pytorch Geometric and DGL dataset objects.
```python
from ogb.graphproppred import GraphPropPredDataset
dataset = GraphPropPredDataset(dataset_name, meta_dict = meta_dict)

# see if it is working properly
print(dataset[0])
print(dataset.get_idx_split())
```

## 9. Zipping and cleaning up
```python
saver.zip()
saver.cleanup()
```

## 10. Sending us two files
In this example, under `submission_ogbg_toy/`, you will find two files `meta_dict.pt` and `toy.zip`. Please send them to us.
