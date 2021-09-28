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

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
                 in_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)
        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
        self.test_idx.share_memory_()

        if self.in_memory:
            self.x = torch.from_numpy(dataset.all_paper_feat).share_memory_()
        else:
            self.x = dataset.paper_feat
        self.y = torch.from_numpy(dataset.all_paper_label)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=4)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=3)

    def convert_batch(self, batch_size, n_id, adjs):
        if self.in_memory:
            x = self.x[n_id].to(torch.float)
        else:
            x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


class GNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_layers: int, heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'gat':
            self.convs.append(
                GATConv(in_channels, hidden_channels // heads, heads))
            self.skips.append(Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels // heads, heads))
                self.skips.append(Linear(hidden_channels, hidden_channels))

        elif self.model == 'graphsage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]
            x = self.convs[i]((x, x_target), adj_t)
            if self.model == 'gat':
                x = x + self.skips[i](x_target)
                x = F.elu(self.norms[i](x))
            elif self.model == 'graphsage':
                x = F.relu(self.norms[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.test_acc(y_hat.softmax(dim=-1), batch.y)
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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='gat',
                        choices=['gat', 'graphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory)

    if not args.evaluate:
        model = GNN(args.model, datamodule.num_features,
                    datamodule.num_classes, args.hidden_channels,
                    num_layers=len(args.sizes), dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode = 'max', save_top_k=1)
        trainer = Trainer(gpus=args.device, max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir=f'logs/{args.model}')
        trainer.fit(model, datamodule=datamodule)

    if args.evaluate:
        dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]

        trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)
        model = GNN.load_from_checkpoint(checkpoint_path=ckpt,
                                         hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res, f'results/{args.model}', mode = 'test-dev')
