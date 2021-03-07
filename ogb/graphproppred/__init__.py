from .evaluate import Evaluator
from .dataset import GraphPropPredDataset

try:
    from .dataset_pyg import PygGraphPropPredDataset
except Exception as e:
    print(e)

try:
    from .dataset_dgl import DglGraphPropPredDataset
    from .dataset_dgl import collate_dgl
except Exception as e:
    print(e)
