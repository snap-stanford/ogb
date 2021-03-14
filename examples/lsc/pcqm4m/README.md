# Baseline code for PCQM4M-LSC

Please refer to our OGB-LSC paper (coming soon) for the detailed setting.

## Installation requirements
```
ogb>=1.3.0
rdkit>=2019.03.1
torch>=1.7.0
```

## Basic commandline arguments
- `LOG_DIR`: Tensorboard log directory.
- `CHECKPOINT_DIR`: Directory to save the best validation checkpoint. The checkpoint file will be saved at `${CHECKPOINT_DIR}/checkpoint.pt`.
- `TEST_DIR`: Directory path to save the test submission. The test file will be saved at `${TEST_DIR}/y_pred_pcqm4m.npz`.

## Baseline models

### GIN [1]
```bash
python main_gnn.py --gnn gin --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### GIN-virtual [1,3]
```bash
python main_gnn.py --gnn gin-virtual --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### GCN [2]
```bash
python main_gnn.py --gnn gcn --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### GCN-virtual [2,3]
```bash
python main_gnn.py --gnn gcn-virtual --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### MLP + Morgan fingerprint baseline [4]
```bash
python main_mlpfp.py --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```


## Performance

| Model              |Valid MAE  | Test MAE*   | \#Parameters    | Hardware |
|:------------------ |:--------------   |:---------------| --------------:|----------|
| GIN     | 0.1536 | 0.1678 | 3.8M  | GeForce RTX 2080 (11GB GPU) |
| GIN-virtual     | 0.1396 | 0.1487 | 6.7M  | GeForce RTX 2080 (11GB GPU) |
| GCN     | 0.1684 | 0.1838 | 2.0M  | GeForce RTX 2080 (11GB GPU) |
| GCN-virtual     | 0.1510 | 0.1579 | 4.9M  | GeForce RTX 2080 (11GB GPU) |
| MLP+Fingerprint     | 0.2044 | 0.2068 | 16.1M  | GeForce RTX 2080 (11GB GPU) |

\* Test MAE is evaluated on the **hidden test set.**

## References
[1] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks?. ICLR 2019

[2] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR 2017

[3] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. Neural message passing for quantum chemistry. ICML 2017.

[4] Morgan, Harry L. "The generation of a unique machine description for chemical structures-a technique developed at chemical abstracts service." Journal of Chemical Documentation 5.2 (1965): 107-113.
