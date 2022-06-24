from .dataset import NodePropPredDataset
from .evaluate import Evaluator

try:
    import torch

    from .dataset_pyg import PygNodePropPredDataset
except ImportError:
    pass

try:
    from .dataset_dgl import DglNodePropPredDataset
except (ImportError, OSError):
    pass
