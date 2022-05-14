# Baseline code for MAG240M

- Please refer to the **[OGB-LSC paper](https://arxiv.org/abs/2103.09430)** for the detailed setting.
- Baseline code based on **[DGL](https://www.dgl.ai/)** is available **[here](https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/MAG240M)**.

## Installation requirements
```
ogb>=1.3.0
torch>=1.7.0
pytorch-lightning>=1.2.0
torch-geometric==master (pip install git+https://github.com/rusty1s/pytorch_geometric.git)
```

## Baseline models

The `MAG240M` dataset will be automatically downloaded to the path denoted in `root.py`.
Please change its content accordingly if you want to download the dataset to a custom hard-drive or folder.
For each experiment, the test submission will be automatically saved in `./results/` after training is done.

We consider a broad range of graph ML models for initial benchmarking efforts in both homogeneous (where only paper to paper relations are considered) and full heterogeneous settings.
Due to the file size of the `MAG240M` node feature matrix, some scripts may require up to 256GB RAM.

### (Graph-Agnostic) MLP

```bash
python mlp.py --device=0
```

### Label Propagation

```bash
python label_prop.py
```

### SGC: Simplified Graph Convolutional Networks [1]

For pre-processing node features, run:

```bash
python preprocess_sgc.py
```

For training the model based on pre-processed node features, run:

```bash
python sgc.py --device=0
```

### SIGN: Scalable Inception Graph Neural Networks [2]

For pre-processing node features, run:

```bash
python preprocess_sgc.py
```

For training the model based on pre-processed node features, run:

```bash
python sign.py --device=0
```

### MLP+C&S: Correct and Smooth [3]

For training graph-agnostic node predictions, run:

```bash
python preprocess_correct_and_smooth.py --device=0
```

For correcting and smoothing graph-agnostic predictions, run:

```bash
python correct_and_smooth.py
```

### GraphSAGE (with Neighbor Sampling) [4]

For training the `GraphSAGE` model, run:

```bash
python gnn.py --device=0 --model=graphsage
```

For evaluating the `GraphSAGE` model on best validation checkpoint, run:

```bash
python gnn.py --device=0 --model=graphsage --evaluate
```

### GAT: Graph Attention Networks (with Neighbor Sampling) [5]

For training the `GAT` model, run:

```bash
python gnn.py --device=0 --model=gat
```

For evaluating the `GAT` model on best validation checkpoint, run:

```bash
python gnn.py --device=0 --model=gat --evaluate
```

### Relational-GraphSAGE (with Neighbor Sampling) [6]

A customized GraphSAGE model that takes the different relation types of the heterogeneous graph into account.
For training the `Relational-GraphSAGE` model, run:

```bash
python rgnn.py --device=0 --model=rgraphsage
```

For evaluating the `Relational-GraphSAGE` model on best validation checkpoint, run:

```bash
python rgnn.py --device=0 --model=rgraphsage --evaluate
```

### Relational-GAT (with Neighbor Sampling) [6]

A customized GAT model that takes the different relation types of the heterogeneous graph into account.
For training the `Relational-GAT` model, run:

```bash
python rgnn.py --device=0 --model=rgat
```

For evaluating the `Relational-GAT` model on best validation checkpoint, run:

```bash
python rgnn.py --device=0 --model=rgat --evaluate
```

## Performance

| Model |Valid Accuracy (%) | Test Accuracy (%)*   | \#Parameters | Hardware |
|:-|:-|:-|:-|:-|
| MLP | 52.67 | 52.73 | 0.5M | GeForce RTX 2080 Ti (11GB GPU) |
| Label Propagation | 58.44 | 56.29 | 0 | --- |
| SGC | 65.82 | 65.29 | 0.7M | GeForce RTX 2080 Ti (11GB GPU) |
| SIGN | 66.64 | 66.09 | 3.8M | GeForce RTX 2080 Ti (11GB GPU) |
| MLP+C&S | 66.98 | 66.18 | 0.5M | GeForce RTX 2080 Ti (11GB GPU) |
| GraphSAGE | 66.79 | 66.28 | 4.9M | GeForce RTX 2080 Ti (11GB GPU) |
| GAT | 67.15 | 66.80 | 4.9M | GeForce RTX 2080 Ti (11GB GPU) |
| R-GraphSAGE | 69.86 | 68.94 | 12.2M | GeForce RTX 2080 Ti (11GB GPU) |
| R-GAT | 70.02 | 69.42 | 12.3M | GeForce RTX 2080 Ti (11GB GPU) |

\* Test Accuracy is evaluated on the **hidden test set.**

## References

[1] Wu *et al.*: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)

[2] Frasca *et al.*: [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198)

[3] Huang *et al.*: [Combining Label Propagation and Simple Models Out-performs Graph Neural Networks](https://arxiv.org/abs/2010.13993)

[4] Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

[5] Veličković *et al.*: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

[6] Schlichtkrull *et al.*: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
