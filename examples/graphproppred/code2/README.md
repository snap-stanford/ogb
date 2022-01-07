# ogbg-code2

This repository includes the scripts for GNN baselines for `ogbg-code2` dataset.
This code requires Pytorch Geometric version>=`2.0.2` and torch version>=`1.10.1`.

**Note (Feb 24, 2021)**: The older version `ogbg-code` is deprecated due to prediction target (i.e., method name) leakage in input AST. `ogbg-code2` (available from `ogb>=1.2.5` ) fixes this issue, where the method name and its recursive definition in AST are replace with a special token `_mask_`. The leaderboard results of `ogbg-code` and `ogbg-code2` are *not* comparable. 

## Training & Evaluation

```
# Run with default config.
# $GNN_TYPE and $FILENAME are described below.
python main_pyg.py --gnn $GNN_TYPE --filename $FILENAME
```

### `$GNN_TYPE`
`$GNN_TYPE` specified the GNN architecture. It should be one of the followings:
- `gin`: GIN [1]
- `gin-virtual`: GIN over graphs augmented with virtual nodes\* [3]
- `gcn`: GCN [2]
- `gin-virtual`: GCN over graphs augmented with virtual nodes\* [3]

\* Additional nodes that are connected to all the nodes in the original graphs.

### `$FILENAME`: Specifying output files. 
`$FILENAME` specifies the filename to save the result. The result is a dictionary containing (1) best training performance (`'BestTrain'`), (2) best validation performance (`'Val'`), (3) test performance at the best validation epoch (`'Test'`), and (4) training performance at the best validation epoch (`'Train'`).


## Converting code snippet into OGB graph object
[`py2graph.py`](https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/code2/py2graph.py) is a script that converts a Python code snippet into a graph object that is fully compatible with the OGB code dataset. 
This script can be used when one wants to perform transfer learning from external Python code datasets.

## References
[1] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks?. ICLR 2019

[2] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR 2017

[3] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. Neural message passing for quantum chemistry. ICML 2017.
