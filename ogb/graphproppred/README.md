# Graph Property Prediction
## 1. Datasets

### `ogbg-mol`: Prediction of chemical properties of molecules
#### - *Dataset description*
A set of molecular prediction datasets adopted from MoleculeNet [1]. All the molecules are pre-processed using RDKit [2].
Each graph represents a molecule, where nodes are atoms and edges are chemical bonds.
Input node features are 9-dimensional, containing atomic number, chirality, formal charge, etc. Input edge features are 3-dimensional, containing bond type, bond stereo, etc.
For further details, please refer to [code](../utils/features.py). 

For encoding these raw input features, we prepare simple modules called `AtomEncoder` and `BondEncoder`. They can be used as follows to embed raw atom and bond features to obtain `atom_emb` and `bond_emb`.
```python
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
atom_encoder = AtomEncoder(emb_dim = 100)
bond_encoder = BondEncoder(emb_dim = 100)

atom_emb = atom_encoder(x) # x is input atom feature in Pytorch Geometric
edge_emb = bond_encoder(edge_attr) # edge_attr is input edge feature in Pytorch Geometric
```

#### - *List of tasks*
There are diverse kinds of chemical properties of molecules, ranging from biological ones from physical ones. 
Successful machine learning models should be able to accurately predict different properties of molecules.
Following MoleculeNet [1], we provide a list of 12 molecule property prediction datasets. The aim is to predict diverse properties of molecules.
Detailed description of each dataset can be found in [1].

*Classification datasets:* `ogbg-mol-bace`, `ogbg-mol-bbbp`, `ogbg-mol-clintox`, `ogbg-mol-muv`, `ogbg-mol-pcba`, `ogbg-mol-sider`, `ogbg-mol-tox21`, `ogbg-mol-toxcast`, `ogbg-mol-hiv`

*Regression datasets:* `ogbg-mol-esol`, `ogbg-mol-freesolv`, `ogbg-mol-lipo`

Each dataset can contain multiple labels/tasks to predict. The goal is to maximize the performance averaged across labels/tasks.


### `ogbg-code`: Prediction of semantic properties of code snippets from their AST graphs
In preparation. Coming soon.

### `ogbg-ppi`: Prediction of species identity from their protein interaction networks
In preparation. Coming soon.



## 2. Modules
### Data loader
We can obtain data-loader for dataset named `d_name` as follows, where we can replace `d_name` with any available datasets (e.g., `"ogbg-mol-tox21"`, `"ogbg-mol-esol"`).

#### - Pytorch Geometric
```python
from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

dataset = PygGraphPropPredDataset(name = d_name) 
num_tasks = dataset.num_tasks # obtaining number of prediction tasks in a dataset
splitted_idx = dataset.get_idx_split() 

train_loader = DataLoader(dataset[splitted_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[splitted_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[splitted_idx["test"]], batch_size=32, shuffle=False)
```
Note that prediction target is stored in `dataset.y`.

#### - DGL
```python
from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader

dataset = DglGraphPropPredDataset(name = d_name)
num_tasks = dataset.num_tasks # obtaining number of prediction tasks in a dataset
splitted_idx = dataset.get_idx_split()

train_loader = DataLoader(dataset[splitted_idx["train"]], batch_size=32, shuffle=True, collate_fn=collate_dgl)
valid_loader = DataLoader(dataset[splitted_idx["valid"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
test_loader = DataLoader(dataset[splitted_idx["test"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
```
Note that prediction target of the $i$-th example can be obtained by e.g., `graph, label = dataset[i]`.

### Evaluator
Evaluators are customized for each dataset.
We require users to pass pre-specified format to the evaluator.
First, please learn the input and output format specification of the evaluator as follows.

```python
from ogb.graphproppred import Evaluator

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
[1] Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K. & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), 513-530. 

[2] Landrum, G. (2006). RDKit: Open-source cheminformatics.

