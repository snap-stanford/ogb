try:
    from .pcqm4m import PCQM4MDataset, PCQM4MEvaluator
except ImportError:
    pass   

try:
    from .pcqm4m_pyg import PygPCQM4MDataset
except ImportError:
    pass

try:
    from .pcqm4m_dgl import DglPCQM4MDataset
except (ImportError, OSError):
    pass

from .mag240m import MAG240MDataset, MAG240MEvaluator
from .wikikg90m import WikiKG90MDataset, WikiKG90MEvaluator
