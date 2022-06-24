from .evaluate import Evaluator
from .dataset import NodePropPredDataset

try:
    from .dataset_pyg import PygNodePropPredDataset
except ImportError:
    pass

try:
    from .dataset_dgl import DglNodePropPredDataset
except (ImportError, OSError):
    pass
