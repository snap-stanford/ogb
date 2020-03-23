from .evaluate import Evaluator
from .dataset import GraphPropPredDataset

try:
    from .dataset_pyg import PygGraphPropPredDataset
except ImportError:
    pass

try:
    from .dataset_dgl import DglGraphPropPredDataset
except ImportError:
    pass
