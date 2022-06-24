from .dataset import GraphPropPredDataset
from .evaluate import Evaluator

try:
    import torch

    from .dataset_pyg import PygGraphPropPredDataset
except ImportError:
    pass

try:
    from .dataset_dgl import DglGraphPropPredDataset, collate_dgl
except (ImportError, OSError):
    pass
