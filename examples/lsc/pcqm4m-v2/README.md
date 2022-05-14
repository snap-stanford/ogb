# Baseline code for PCQM4Mv2

- Please refer to the **[OGB-LSC paper](https://arxiv.org/abs/2103.09430)** for the detailed setting.
- Baseline code based on **[DGL](https://www.dgl.ai/)** is available **[here](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/PCQM4M)**.

## Installation requirements
```
ogb>=1.3.2
rdkit>=2021.03.1
torch>=1.7.0
```

## Basic commandline arguments
- `LOG_DIR`: Tensorboard log directory.
- `CHECKPOINT_DIR`: Directory to save the best validation checkpoint. The checkpoint file will be saved at `${CHECKPOINT_DIR}/checkpoint.pt`.
- `TEST_DIR`: Directory path to save the test submission. The test file will be saved at `${TEST_DIR}/y_pred_pcqm4mv2.npz`.

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

## Measuring the Test Inference Time
The code below takes **the raw SMILES strings as input**, uses the saved checkpoint, and perform inference over for all the 147,037 test-dev (147,432 test-challenge) molecules.
```bash
python test_inference_gnn.py --gnn $GNN --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

For GIN-virtual, the total inference (from SMILES strings to target values) takes around 1 minute on a single GeForce RTX 2080 GPU and an Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz.
For your model, **the total inference time needs to be less than 4 hours on a single GPU and a CPU**. Ideally, you should use the GPU/CPU with the same spec as ours. However, we also allow the use of other GPU/CPU specs, as long as the specs are clearly reported in the final submission.

## Performance

| Model              |Valid MAE  | Test-dev MAE*   | \#Parameters    | Hardware |
|:------------------ |:--------------   |:---------------| --------------:|----------|
| GIN     | 0.1195 | 0.1218 | 3.8M  | GeForce RTX 2080 (11GB GPU) |
| GIN-virtual     | 0.1083 | 0.1084 | 6.7M  | GeForce RTX 2080 (11GB GPU) |
| GCN     | 0.1379 | 0.1398 | 2.0M  | GeForce RTX 2080 (11GB GPU) |
| GCN-virtual     | 0.1153 | 0.1152 | 4.9M  | GeForce RTX 2080 (11GB GPU) |
| MLP+Fingerprint     | 0.1753 | 0.1760 | 16.1M  | GeForce RTX 2080 (11GB GPU) |

\* Test MAE is evaluated on the **hidden test-dev set.**

## 3D graphs

We further provide the equilibrium 3D graph structure for training molecules. The zipped folder can be downloaded **[here](http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip)** (2.7GB). The folder contains the xyz coordinate files of all the training molecules. For `i`-th molecule, the corresponding xyz file is `i.xyz`, e.g., xyz file of the 1234-th molecule is named `1234.xyz`. The community should feel free to exploit 3D structural information to improve their model performance. Note that 3D information is *not* provided for validation and test molecules, and test-time inference needs to be performed without explicit 3D information.

## References
[1] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks?. ICLR 2019

[2] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR 2017

[3] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. Neural message passing for quantum chemistry. ICML 2017.

[4] Morgan, Harry L. "The generation of a unique machine description for chemical structures-a technique developed at chemical abstracts service." Journal of Chemical Documentation 5.2 (1965): 107-113.
