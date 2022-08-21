# Baseline code for OGBL-VESSEL

This is an external contribution to OGB, please refer to the **[NeuRIPS paper](https://arxiv.org/abs/2108.13233)** for the detailed setting.

## Installation requirements
```
ogb>=1.3.4
torch>=1.7.0
torch-geometric==master (pip install git+https://github.com/rusty1s/pytorch_geometric.git)
```

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/vessel/mlp.py)**: Full-batch MLP training based on Node2Vec features. This script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python node2vec.py` [requires `torch-geometric>=1.5.0`].
* **[GNN](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/vessel/gnn.py)**: Full-batch GNN training using either the GCN or GraphSAGE operator (`--use_sage`) [requires `torch-geometric>=1.6.0`].
* **[Matrix Factorization](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/vessel/mf.py)**: Full-batch Matrix Factorization training.

## Hyperparameter Search

In order to select the values, we employed Grid Search using wandb.ai based sweeps (c.f. yaml-configuration files). Due to the huge size of our dataset, hyperparameter tuning is challenging. To overcome this challenge, we subsample a region of the mouse brain in order to create a small graph. To ensure we are not introducing any bias, we measured the KL-Divergence to ensure that our small graph is representative of the whole brain in its distribution of vasculature. We selected the best set of hyperparameters on the small graph and used it on the actual graph with small modifications if needed.

## Training & Evaluation

```
# Run with default config
python gnn.py

# Run with custom config
python gnn.py --hidden_channels=128

# Use Node2Vec embeddings
python gnn.py --hidden_channels=128 --use_node_embedding
```
## Sampling Strategy

Following the NeurIPS paper, we propose to use a spatial sampling strategy that constitutes a more challening and realistic scenario in comparison to random sampling.
Random sampling ultimately creates unrealistic edges, i.e. vessels connecting totally different brain regions or unrealistic structures. These are very easy to identify for a classifier,
leading to overly optimistic scores on the link prediction task. In contrast, spatial sampling (as proposed in our paper) creates more challenging edges, and biologically more realistic structures. These are harder to differentiate from the (actual) edges in the whole brain graph. Employing the spatial sampling criteria, we force the link predictor to learn more meaningful representations that will lead to biologically accurate results when using the link prediction algorithm for missing link prediction or graph completion.
We kindly ask you to employ the spatial sampling structure for negative edges by making use of the presampled negative edges.

## Performance

| Model |Highest Valid Accuracy (%) | Final Test Accuracy (%)  | #num_params | Hardware |
|:-|:-|:-|:-|:-|
| MF |  49.99 ± 0.06 | 49.97 ± 0.05 | 8641| GeForce Quadro RTX 8000 Ti (48GB GPU) |
| MLP |  48.01 ± 1.32 | 47.94 ± 1.33 | 1037577 | GeForce Quadro RTX 8000 Ti (48GB GPU) |
| GCN | 43.49 ± 9.61 | 43.53 ± 9.61 | 396289 | GeForce Quadro RTX 8000 Ti (48GB GPU) |
| GraphSAGE|  49.93 ± 6.76 | 49.89 ± 6.78 | 396289 | GeForce Quadro RTX 8000 Ti (48GB GPU) |
| GCN + Node2Vec|49.60 ± 0.61 | 49.54 ± 0.57| 226744513| GeForce Quadro RTX 8000 Ti (48GB GPU) |
| GraphSAGE + Node2VEc| 47.36 ± 1.36| 47.35 ± 1.36| 226892737| GeForce Quadro RTX 8000 Ti (48GB GPU) |

## Citing

Please consider citing this work if any of our code or dataset is helpful for your research. Considering the specific graphs and baseline models please also cite the respective original articles as described in the preprint.

- [arXiv link](https://arxiv.org/abs/2108.13233)
- [Published in NIPS 2021 Dataset & Benchmark Track](https://nips.cc/Conferences/2021/ScheduleMultitrack?event=29873)

```
@misc{paetzold2021brain,
      title={Whole Brain Vessel Graphs: A Dataset and Benchmark for Graph Learning and Neuroscience (VesselGraph)}, 
      author={Johannes C. Paetzold and Julian McGinnis and Suprosanna Shit and Ivan Ezhov and Paul Büschl and Chinmay Prabhakar and Mihail I. Todorov and Anjany Sekuboyina and Georgios Kaissis and Ali Ertürk and Stephan Günnemann and Bjoern H. Menze},
      year={2021},
      eprint={2108.13233},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
