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

try:
    from .pcqm4mv2 import PCQM4Mv2Dataset, PCQM4Mv2Evaluator
except ImportError:
    pass   

try:
    from .pcqm4mv2_pyg import PygPCQM4Mv2Dataset
except ImportError:
    pass

try:
    from .pcqm4mv2_dgl import DglPCQM4Mv2Dataset
except (ImportError, OSError):
    pass

from .mag240m import MAG240MDataset, MAG240MEvaluator
from .wikikg90m import WikiKG90MDataset, WikiKG90MEvaluator
from .wikikg90mv2 import WikiKG90Mv2Dataset, WikiKG90Mv2Evaluator
