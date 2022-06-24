from .evaluate import Evaluator
from .dataset import LinkPropPredDataset

try:
    from .dataset_pyg import PygLinkPropPredDataset
except ImportError:
    pass

try:
    from .dataset_dgl import DglLinkPropPredDataset
except (ImportError, OSError):
    pass
