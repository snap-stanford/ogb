from .evaluate import Evaluator
from .dataset import GraphPropPredDataset

try:
    from .dataset_pyg import PygGraphPropPredDataset
except ImportError:
    pass

try:
    from .dataset_dgl import DglGraphPropPredDataset
    from .dataset_dgl import collate_dgl
except (ImportError, OSError):
    pass
