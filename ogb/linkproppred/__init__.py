from .evaluate import Evaluator
from .dataset import LinkPropPredDataset

try:
    from .dataset_pyg import PygLinkPropPredDataset
except Exception as e:
    print(e)

try:
    from .dataset_dgl import DglLinkPropPredDataset
except Exception as e:
    print(e)
