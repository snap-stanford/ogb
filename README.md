<p align="center">
  <img width="40%" src="https://snap-stanford.github.io/ogb-web/assets/img/OGB_rectangle.png" />
</p>

--------------------------------------------------------------------------------

[![PyPI](https://img.shields.io/pypi/v/ogb)](https://pypi.org/project/ogb/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/snap-stanford/ogb/blob/master/LICENSE)

## Overview

The Open Graph Benchmark (OGB) is a collection of benchmark datasets, data loaders, and evaluators for graph machine learning. Datasets cover a variety of graph machine learning tasks and real-world applications.
The OGB data loaders are fully compatible with popular graph deep learning frameworks, including [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and [Deep Graph Library (DGL)](https://www.dgl.ai/). They provide automatic dataset downloading, standardized dataset splits, and unified performance evaluation.

<p align="center">
  <img width="80%" src="https://snap-stanford.github.io/ogb-web/assets/img/ogb_overview.png" />
</p>

OGB aims to provide graph datasets that cover important graph machine learning tasks, diverse dataset scale, and rich domains.

**Graph ML Tasks:** We cover three fundamental graph machine learning tasks: prediction at the level of nodes, links, and graphs.

**Diverse scale:** Small-scale graph datasets can be processed within a single GPU, while medium- and large-scale graphs might require multiple GPUs or clever sampling/partition techniques.

**Rich domains:** Graph datasets come from diverse domains ranging from scientific ones to social/information networks, and also include heterogeneous knowledge graphs. 

<p align="center">
  <img width="70%" src="https://snap-stanford.github.io/ogb-web/assets/img/dataset_overview.png" />
</p>

OGB is an on-going effort, and we are planning to increase our coverage in the future.

## Installation
You can install OGB using Python's package manager `pip`.

#### Requirements
 - Python>=3.5
 - PyTorch>=1.2
 - DGL>=0.4.1 or torch-geometric>=1.3.1
 - Numpy>=1.16.0
 - pandas>=0.24.0
 - urllib3>=1.24.0
 - scikit-learn>=0.20.0

#### Pip install
The recommended way to install OGB is using Python's package manager pip:
```bash
pip install ogb
```

#### From source
You can also install OGB from source. This is recommended if you want to contribute to OGB.
```bash
git clone https://github.com/snap-stanford/ogb
cd ogb
python setup.py install
```

## Example
We highlight two key features of OGB, namely, (1) easy-to-use data loaders, and (2) standardized evaluators.
#### (1) Data loaders
We prepare easy-to-use PyTorch Geometric and DGL data loaders. We handle dataset downloading as well as standardized dataset splitting.
Below, on PyTorch Geometric, we see that a few lines of code is sufficient to prepare and split the dataset! Needless to say, you can enjoy the same convenience for DGL!
```python
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

dataset = PygGraphPropPredDataset(name = "ogbg-mol-hiv")

splitted_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[splitted_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[splitted_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[splitted_idx["test"]], batch_size=32, shuffle=False)
```

#### (2) Evaluators
We also prepare standardized evaluators for easy evaluation and comparison of different methods. The evaluator takes `input_dict` (a dictionary whose format is specified in `evaluator.expected_input_format`) as input, and returns a dictionary storing the performance metric appropriate for the given dataset.
The standardized evaluation protocol allows researchers to reliably compare their methods.
```python
from ogb.graphproppred import Evaluator

evaluator = Evaluator(name = "ogbg-mol-hiv")
# You can learn the input and output format specification of the evaluator as follows.
# print(evaluator.expected_input_format) 
# print(evaluator.expected_output_format) 
input_dict = {"y_true": y_true, "y_pred": y_pred}
result_dict = evaluator.eval(input_dict) # E.g., {"rocauc": 0.7321}
```

## Citing OGB
Coming soon.
