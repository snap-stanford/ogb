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
import pathlib
from torch.profiler import ProfilerActivity, profile

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
        node_types = ['paper', 'author', 'institution']
        edge_types = [
            ('author', 'affiliated_with', 'institution'),
            ('institution', 'rev_affiliated_with', 'author'),
            ('author', 'writes', 'paper'),
            ('paper', 'rev_writes', 'author'),
            ('paper', 'cites', 'paper'),
        ]
        return node_types, edge_types        

class GNN(torch.nn.Module):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_layers: int, heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.model = model.lower()
        self.dropout = dropout
        self.num_layers = num_layers

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = x.to(torch.float)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

class HeteroGNN(LightningModule):
    def __init__(self, model_name: str, metadata: Tuple[List[NodeType], List[EdgeType]], in_channels: int, out_channels: int,
                 hidden_channels: int, num_layers: int, heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        model = GNN(model_name, in_channels, out_channels, hidden_channels, num_layers, heads=heads, dropout=dropout)
        self.model = to_hetero(model, metadata, aggr='sum', debug=True).to(device)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        return self.model(x_dict, edge_index_dict)

    def common_step(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        batch_size = batch['paper'].batch_size
        y_hat = self(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        y = batch['paper'].y[:batch_size].to(torch.long)
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

    def predict_step(self, batch: Batch, batch_idx: int):
        y_hat, y = self.common_step(batch)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]

def trace_handler(p):
    if torch.cuda.is_available():
        profile_sort = 'self_cuda_time_total'
    else:
        profile_sort = 'self_cpu_time_total'
    output = p.key_averages().table(sort_by=profile_sort)
    print(output)
    profile_dir = str(pathlib.Path.cwd()) + '/'
    timeline_file = profile_dir + 'timeline' + '.json'
    p.export_chrome_trace(timeline_file)

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
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    dataset = MAG240MDataset(ROOT)
    data = dataset.to_pyg_hetero_data()
    datamodule = MAG240M(data, ('paper', data['paper'].train_mask),
                        ('paper', data['paper'].val_mask),
                        ('paper', data['paper'].test_mask),
                        ('paper', data['paper'].test_mask),
                        loader='neighbor', num_neighbors=args.sizes,
                        batch_size=args.batch_size, num_workers=2)
    print(datamodule)

    if not args.evaluate:
        model = HeteroGNN(args.model, datamodule.metadata(), datamodule.num_features,
                    datamodule.num_classes, args.hidden_channels,
                    num_layers=len(args.sizes), dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode = 'max', save_top_k=1)
        trainer = Trainer(accelerator="cpu", max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir=f'logs/{args.model}',
                          limit_train_batches=10, limit_test_batches=10,
                          limit_val_batches=10, limit_predict_batches=10)
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

        trainer.predict(model=model, datamodule=datamodule)
        if args.profile:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            on_trace_ready=trace_handler) as p:
                trainer.predict(model=model, datamodule=datamodule)
                p.step()

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
