# ogbg-mol

This repository includes the scripts for GNN baselines for `ogbg-mol*` dataset.

## Training & Evaluation

```
# Run with default config.
# $DATASET, $GNN_TYPE, and $FILENAME are described below.
python main_pyg.py --dataset $DATASET --gnn $GNN_TYPE --filename $FILENAME
```

### `$DATASET`
`$DATASET` specified the name of the molecule dataset. It should be one of the followings:
- `ogbg-molhiv`
- `ogbg-molpcba`

Additionally we provide the smaller molecule datasets from MoleculeNet [1].
- `ogbg-molbace`
- `ogbg-molbbbp`
- `ogbg-molclintox`
- `ogbg-molmuv`
- `ogbg-molsider`
- `ogbg-moltox21`
- `ogbg-moltoxcast`
- `ogbg-molesol`
- `ogbg-molfreesolv`
- `ogbg-mollipo`

The last three datasets (`ogbg-molesol`, `ogbg-molfreesolv`, `ogbg-mollipo`) are for regression, and the rest are for binary classification.

### `$GNN_TYPE`
`$GNN_TYPE` specified the GNN architecture. It should be one of the followings:
- `gin`: GIN [2]
- `gin-virtual`: GIN over graphs augmented with virtual nodes\* [4]
- `gcn`: GCN [3]
- `gcn-virtual`: GCN over graphs augmented with virtual nodes\* [4]

\* Additional nodes that are connected to all the nodes in the original graphs.

### `$FILENAME`: Specifying output files. 
`$FILENAME` specifies the filename to save the result. The result is a dictionary containing (1) best training performance (`'BestTrain'`), (2) best validation performance (`'Val'`), (3) test performance at the best validation epoch (`'Test'`), and (4) training performance at the best validation epoch (`'Train'`).

## Converting SMILES string into OGB graph object
Molecules are typically represented as [SMILES strings](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system).
OGB package provides the utility to transform the SMILES string into a graph object that is fully compatible with the OGB molecule datasets. This can be used when one wants to perform transfer learning from external molecular datasets.

```python
from ogb.utils import smiles2graph
graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
```

## References
[1] Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., ... & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), 513-530.

[2] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks?. ICLR 2019

[3] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR 2017

[4] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. Neural message passing for quantum chemistry. ICML 2017.
