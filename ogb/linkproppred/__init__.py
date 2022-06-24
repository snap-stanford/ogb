from .dataset import LinkPropPredDataset
from .evaluate import Evaluator

try:
    import torch

    from .dataset_pyg import PygLinkPropPredDataset
except ImportError:
    pass

try:
    from .dataset_dgl import DglLinkPropPredDataset
except (ImportError, OSError):
    pass
