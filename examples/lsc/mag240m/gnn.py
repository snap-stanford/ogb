import os
import time
import glob
import argparse
import os.path as osp
from tqdm import tqdm

from typing import Optional, List, NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR

from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT
from torch_geometric.loader.neighbor_loader import NeighborLoader
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
from torch_geometric.typing import EdgeType, NodeType
from typing import Dict, Tuple
from torch_geometric.data import Batch
from torch_geometric.data import LightningNodeData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MAG240M(LightningNodeData):
    def __init__(self, *args, **kwargs):
        super(MAG240M, self).__init__(*args, **kwargs)

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    def metadata(self) -> Tuple[List[NodeType], List[EdgeType]]:
        node_types = ['paper']
        edge_types = [('author', 'affiliated_with', 'institution'),
                      ('author', 'writes', 'paper'),
                      ('paper', 'cites', 'paper')]
        return node_types, edge_types        

class GNN(torch.nn.Module):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_layers: int, heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.model = model.lower()
        self.dropout = dropout
        self.num_layers = num_layers

        # self.convs = ModuleList()
        # self.norms = ModuleList()
        # self.skips = ModuleList()

        # if self.model == 'gat':
        #     self.convs.append(
        #         GATConv(in_channels, hidden_channels // heads, heads))
        #     self.skips.append(Linear(in_channels, hidden_channels))
        #     for _ in range(num_layers - 1):
        #         self.convs.append(
        #             GATConv(hidden_channels, hidden_channels // heads, heads))
        #         self.skips.append(Linear(hidden_channels, hidden_channels))

        # elif self.model == 'graphsage':
        #     self.convs.append(SAGEConv(in_channels, hidden_channels))
        #     for _ in range(num_layers - 1):
        #         self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # for _ in range(num_layers):
        #     self.norms.append(BatchNorm1d(hidden_channels))

        # self.mlp = Sequential(
        #     Linear(hidden_channels, hidden_channels),
        #     BatchNorm1d(hidden_channels),
        #     ReLU(inplace=True),
        #     Dropout(p=self.dropout),
        #     Linear(hidden_channels, out_channels),
        # )
        
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # for i in range(self.num_layers):
        #     x = self.convs[i](x, edge_index)
        #     if self.model == 'gat':
        #         x = x + self.skips[i](x)
        #         x = F.elu(self.norms[i](x))
        #     elif self.model == 'graphsage':
        #         x = F.relu(self.norms[i](x))
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        # return self.mlp(x)

class HeteroGNN(LightningModule):
    def __init__(self, model_name: str, metadata: Tuple[List[NodeType], List[EdgeType]], in_channels: int, out_channels: int,
                 hidden_channels: int, num_layers: int, heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        model = GNN(model_name, in_channels, out_channels, hidden_channels, num_layers, heads=heads, dropout=dropout)
        self.model = to_hetero(model, metadata, aggr='sum').to(device)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        return self.model(x_dict, edge_index_dict)

    # @torch.no_grad()
    # def setup(self, stage: Optional[str] = None):  # Initialize parameters.
    #     data = self.trainer.datamodule
    #     loader = data.dataloader(torch.arange(1), shuffle=False, num_workers=0)
    #     batch = next(iter(loader))
    #     self(batch.x_dict, batch.edge_index_dict)

    def common_step(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        batch_size = batch['paper'].batch_size
        y_hat = self(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        y = batch['paper'].y[:batch_size]
        return y_hat, y

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        y_hat, y = self.common_step(batch)
        train_loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch: Batch, batch_idx: int):
        y_hat, y = self.common_step(batch)
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch: Batch, batch_idx: int):
        y_hat, y = self.common_step(batch)
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model', type=str, default='gat',
                        choices=['gat', 'graphsage'])
    parser.add_argument('--sizes', type=str, default='2')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    dataset = MAG240MDataset(ROOT)
    data = dataset.to_pyg_hetero_data()
    datamodule = MAG240M(data, loader='neighbor', num_neighbors=args.sizes, batch_size=args.batch_size, num_workers=2)
    print(datamodule)

    if not args.evaluate:
        model = HeteroGNN(args.model, datamodule.metadata(), datamodule.num_features,
                    datamodule.num_classes, args.hidden_channels,
                    num_layers=len(args.sizes), dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode = 'max', save_top_k=1)
        trainer = Trainer(accelerator="cpu", max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir=f'logs/{args.model}')
        trainer.fit(model, datamodule=datamodule)

    if args.evaluate:
        dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]

        trainer = Trainer(accelerator="cpu", resume_from_checkpoint=ckpt)
        model = HeteroGNN.load_from_checkpoint(checkpoint_path=ckpt,
                                         hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        trainer.test(model=model, datamodule=datamodule)

        # evaluator = MAG240MEvaluator()
        # loader = datamodule.hidden_test_dataloader()

        # model.eval()
        # device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        # model.to(device)
        # y_preds = []
        # for batch in tqdm(loader):
        #     batch = batch.to(device)
        #     with torch.no_grad():
        #         out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
        #         y_preds.append(out)
        # res = {'y_pred': torch.cat(y_preds, dim=0)}
        # evaluator.save_test_submission(res, f'results/{args.model}', mode = 'test-dev')
