# ogbg-ppa

This repository includes the scripts for GNN baselines for `ogbg-ppa` dataset.

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


## References
[1] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks?. ICLR 2019

[2] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR 2017

[3] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. Neural message passing for quantum chemistry. ICML 2017.
