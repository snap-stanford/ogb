## Built-in Datasets and Benchmark

DGL-KE provides five built-in knowledge graphs:

| Dataset | #nodes | #edges | #relations |
|---------|--------|--------|------------|
| [FB15k](https://data.dgl.ai/dataset/FB15k.zip) | 14951 | 592213 | 1345 |
| [FB15k-237](https://data.dgl.ai/dataset/FB15k-237.zip) | 14541 | 310116 | 237 |
| [wn18](https://data.dgl.ai/dataset/wn18.zip) | 40943 | 151442 | 18 |
| [wn18rr](https://data.dgl.ai/dataset/wn18rr.zip) | 40943 | 93003 | 11 |
| [Freebase](https://data.dgl.ai/dataset/Freebase.zip) | 86054151 | 338586276 | 14824 |

Users can specify one of the datasets with `--dataset` option in their tasks.

## Benchmark result

DGL-KE also provides benchmark results on `FB15k`, `wn18`, as well as `Freebase`. Users can go to the corresponded folder to check out the scripts and results. All the benchmark results are done by AWS EC2. For multi-cpu and distributed training, the target instance is `r5dn.24xlarge`, which has 48 CPU cores and 768 GB memory. Also, `r5dn.xlarge` has 100Gbit network throughput, which is powerful for distributed training. For GPU training, our target instance is `p3.16xlarge`, which has 64 CPU cores and 8 Nvidia v100 GPUs. For users, you can choose your own instance by your demand and tune the hyper-parameters for the best performance.

### FB15k

#### One-GPU training

|  Models    |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|------------|-------|-------|--------|--------|---------|------|
| TransE_l1  | 47.34 | 0.672 | 0.557  | 0.763  | 0.849   | 201  |
| TransE_l2  | 47.04 | 0.649 | 0.525  | 0.746  | 0.844   | 167  |
| DistMult   | 61.43 | 0.696 | 0.586  | 0.782  | 0.873   | 150  |
| ComplEx    | 64.73 | 0.757 | 0.672  | 0.826  | 0.886   | 171  |
| RESCAL     | 124.5 | 0.661 | 0.589  | 0.704  | 0.787   | 1252 |
| TransR     | 59.99 | 0.670 | 0.585  | 0.728  | 0.808   | 530  |
| RotatE     | 43.85 | 0.726 | 0.632  | 0.799  | 0.873   | 1405 |
| SimplE     | 58.85 | 0.709 | 0.619  | 0.773  | 0.862   | 194  |

#### 8-GPU training

|  Models    |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|------------|-------|-------|--------|--------|---------|------|
| TransE_l1  | 48.59 | 0.662 | 0.542  | 0.756  |  0.846  | 53   |
| TransE_l2  | 47.52 | 0.627 | 0.492  | 0.733  |  0.838  | 49   |
| DistMult   | 59.44 | 0.679 | 0.566  | 0.764  |  0.864  | 47   |
| ComplEx    | 64.98 | 0.750 | 0.668  | 0.814  |  0.883  | 49   |
| RESCAL     | 133.3 | 0.643 | 0.570  | 0.685  |  0.773  | 179  |
| TransR     | 66.51 | 0.666 | 0.581  | 0.724  |  0.803  | 90   |
| RotatE     | 50.04 | 0.685 | 0.581  | 0.763  |  0.851  | 120  |
| SimplE     | 64.74 | 0.743 | 0.666  | 0.797  |  0.873  | 72   |

#### Multi-CPU training

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|-----------|-------|-------|--------|--------|---------|------|
| TransE_l1 | 48.32 | 0.645 | 0.521  | 0.741  |  0.838  | 140  |
| TransE_l2 | 45.28 | 0.633 | 0.501  | 0.735  |  0.840  | 58   |
| DistMult  | 62.63 | 0.647 | 0.529  | 0.733  |  0.846  | 58   |
| ComplEx   | 67.83 | 0.694 | 0.590  | 0.772  |  0.863  | 69   |

#### Distributed training

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|-----------|-------|-------|--------|--------|---------|------|
| TransE_l1 | 38.26 | 0.691 | 0.591  | 0.765  |  0.853  | 104  |
| TransE_l2 | 34.84 | 0.645 | 0.510  | 0.754  |  0.854  | 31   |
| DistMult  | 51.85 | 0.661 | 0.532  | 0.762  |  0.864  | 57   |
| ComplEx   | 62.52 | 0.667 | 0.567  | 0.737  |  0.836  | 65   |


### wn18

#### One-GPU training

|  Models    |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|------------|-------|-------|--------|--------|---------|------|
| TransE_l1  | 355.4 | 0.764 | 0.602  | 0.928  |  0.949  | 327  |
| TransE_l2  | 209.4 | 0.560 | 0.306  | 0.797  |  0.943  | 223  |
| DistMult   | 419.0 | 0.813 | 0.702  | 0.921  |  0.948  | 133  |
| ComplEx    | 318.2 | 0.932 | 0.914  | 0.948  |  0.959  | 144  |
| RESCAL     | 563.6 | 0.848 | 0.792  | 0.898  |  0.928  | 308  |
| TransR     | 432.8 | 0.609 | 0.452  | 0.736  |  0.850  | 906  |
| RotatE     | 451.6 | 0.944 | 0.940  | 0.945  |  0.950  | 671  |
| SimplE     | 370.2 | 0.938 | 0.925  | 0.949  |  0.956  | 151  |

#### 8-GPU training

|  Models    |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|------------|-------|-------|--------|--------|---------|------|
| TransE_l1  | 348.8 | 0.739 | 0.553  | 0.927  | 0.948   | 111  |
| TransE_l2  | 198.9 | 0.559 | 0.305  | 0.798  | 0.942   | 71   |
| DistMult   | 798.8 | 0.806 | 0.705  | 0.903  | 0.932   | 66   |
| ComplEx    | 535.0 | 0.938 | 0.931  | 0.944  | 0.949   | 53   |
| RotatE     | 487.7 | 0.943 | 0.939  | 0.945  | 0.951   | 127  |
| SimplE     | 513.4 | 0.945 | 0.940  | 0.950  | 0.953   | 121  |
#### Multi-CPU training

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|-----------|-------|-------|--------|--------|---------|------|
| TransE_l1 | 376.3 | 0.593 | 0.264  | 0.926  | 0.949   | 925  |
| TransE_l2 | 218.3 | 0.528 | 0.259  | 0.777  | 0.939   | 210  |
| DistMult  | 837.4 | 0.791 | 0.675  | 0.904  | 0.933   | 362  |
| ComplEx   | 806.3 | 0.904 | 0.881  | 0.926  | 0.937   | 281  |

#### Distributed training

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|-----------|-------|-------|--------|--------|---------|------|
| TransE_l1 | 136.0 | 0.848 | 0.768  | 0.927  | 0.950   | 759  |
| TransE_l2 | 85.04 | 0.797 | 0.672  | 0.921  | 0.958   | 144  |
| DistMult  | 278.5 | 0.872 | 0.816  | 0.926  | 0.939   | 275  |
| ComplEx   | 333.8 | 0.838 | 0.796  | 0.870  | 0.906   | 273  |

### Freebase

#### 8-GPU training

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|-----------|-------|-------|--------|--------|---------|------|
| TransE_l2 | 23.56 | 0.736 |  0.663 | 0.782  | 0.873   | 4767 |
| DistMult  | 46.19 | 0.833 |  0.813 | 0.842  | 0.869   | 4281 |
| ComplEx   | 46.70 | 0.834 |  0.815 | 0.843  | 0.869   | 8356 |
| TransR    | 49.68 | 0.696 |  0.653 | 0.716  | 0.773   |14235 |
| RotatE    | 93.20 | 0.769 |  0.748 | 0.779  | 0.804   | 9060 |

#### Multi-CPU training

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|-----------|-------|-------|--------|--------|---------|------|
| TransE_l2 | 30.82 | 0.815 |  0.766 | 0.848  | 0.902   | 6993 |
| DistMult  | 44.16 | 0.834 |  0.815 | 0.843  | 0.869   | 7146 |
| ComplEx   | 45.62 | 0.835 |  0.817 | 0.843  | 0.870   | 8732 |

#### Distributed training

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|-----------|-------|-------|--------|--------|---------|------|
| TransE_l2 | 34.25 | 0.764 | 0.705  | 0.802  | 0.869   | 1633 |
| DistMult  | 75.15 | 0.769 | 0.751  | 0.779  | 0.801   | 1679 |
| ComplEx   | 77.83 | 0.771 | 0.754  | 0.779  | 0.802   | 2293 |

### loss function comparison with 8-GPU training
#### FB15k
|  Models   |  Loss      | Pairwise | MR     |  MRR  | HITS@1 | HITS@3 | HITS@10 | TIME |
|-----------|------------|----------|--------|-------|--------|--------|---------|------|
| DistMult  | Hinge      | False    | 111.75 | 0.775 | 0.709  | 0.827  | 0.883   | 59   |
| DistMult  | Hinge      | True     | 100.29 | 0.539 | 0.411  | 0.619  | 0.767   | 59   |
| DistMult  | Logistic   | False    | 57.09  | 0.690 | 0.578  | 0.773  | 0.873   | 56   |
| DistMult  | Logistic   | True     | 74.25  | 0.385 | 0.271  | 0.467  | 0.602   | 58   |
| DistMult  | Logsigmoid | False    | 57.77  | 0.687 | 0.577  | 0.772  | 0.869   | 60   |
| TransE_l1 | Hinge      | False    | 44.74  | 0.703 | 0.607  | 0.778  | 0.855   | 69   |
| TransE_l1 | Hinge      | True     | 69.58  | 0.488 | 0.283  | 0.653  | 0.796   | 64   |
| TransE_l1 | Logistic   | False    | 42.39  | 0.669 | 0.551  | 0.761  | 0.851   | 65   |
| TransE_l1 | Logistic   | True     | 63.34  | 0.488 | 0.337  | 0.593  | 0.739   | 66   |
| TransE_l1 | Logsigmoid | False    | 43.11  | 0.668 | 0.549  | 0.760  | 0.852   | 67   |
| RotatE    | Hinge      | False    | 64.33  | 0.692 | 0.605  | 0.751  | 0.837   | 130  |
| RotatE    | Hinge      | True     | 80.89  | 0.602 | 0.485  | 0.683  | 0.797   | 131  |
| RotatE    | Logistic   | False    | 39.24  | 0.719 | 0.629  | 0.785  | 0.867   | 132  |
| RotatE    | Logistic   | True     | 63.72  | 0.586 | 0.466  | 0.666  | 0.789   | 128  |
| RotatE    | Logsigmoid | False    | 39.42  | 0.720 | 0.631  | 0.786  | 0.865   | 127  |

