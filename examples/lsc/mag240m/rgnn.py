import os
import time
import glob
import argparse
import os.path as osp
from tqdm import tqdm
import traceback

from typing import Optional, List, NamedTuple

import numpy as np
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


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    print('Performing memory-efficient column slicing...')
    chunk = 100000
    return_mat_list = []
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        end_idx = min(i + chunk, end_row_idx)
        tmp_arr = x[i:end_idx, start_col_idx:end_col_idx]
        return_mat_list.append(tmp_arr.copy())
        del tmp_arr

    return np.concatenate(return_mat_list, axis=0)


def save_col_slice(x_from, x_to, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_from.shape[0] == end_row_idx - start_row_idx
    assert x_from.shape[1] == end_col_idx - start_col_idx

    chunk = 100000
    print('Memory-efficient writing...')
    for i, j in tqdm(
            list(
                zip(range(start_row_idx, end_row_idx, chunk),
                    range(0, end_row_idx - start_row_idx, chunk)))):
        end_idx_i = min(i + chunk, end_row_idx)
        end_idx_j = min(j + chunk, end_row_idx - start_row_idx)
        x_to[i:end_idx_i, start_col_idx:end_col_idx] = x_from[j:end_idx_j]


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_relations(self) -> int:
        return 5

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)

        path = f'{dataset.root}/mag240m/paper_to_paper_symmetric.pt'
        if not osp.exists(path):  # Will take approximately 5 minutes...
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

        path = f'{dataset.root}/mag240m/full_adj_t.pt'
        if not osp.exists(path):  # Will take approximately 16 minutes...
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(
                f'{dataset.root}/mag240m/paper_to_paper_symmetric.pt').coo()
            rows, cols = [row], [col]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                      sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.root}/mag240m/full_feat.npy'
        # indicate whether full_feat processing has been finished or not
        done_flag_path = f'{dataset.root}/mag240m/full_feat_done.txt'
        if not osp.exists(
                done_flag_path):  # Will take approximately 3 hours...
            if os.path.exists(path):
                print('Removing unfinished full_feat.npy')
                os.remove(path)

            try:
                t = time.perf_counter()
                print('Generating full feature matrix...')

                N = (dataset.num_papers + dataset.num_authors +
                     dataset.num_institutions)

                x = np.memmap(path, dtype=np.float16, mode='w+',
                              shape=(N, self.num_features))
                paper_feat = dataset.paper_feat
                dim_chunk = 64
                chunk = 100000

                print('Copying paper features...')
                for i in tqdm(range(0, dataset.num_papers,
                                    chunk)):  # Copy paper features.
                    end_idx = min(i + chunk, dataset.num_papers)
                    x[i:end_idx] = paper_feat[i:end_idx]

                edge_index = dataset.edge_index('author', 'writes', 'paper')
                row, col = torch.from_numpy(edge_index)
                adj_t = SparseTensor(
                    row=row, col=col,
                    sparse_sizes=(dataset.num_authors, dataset.num_papers),
                    is_sorted=True)

                print('Generating author features...')
                # processing 64-dim subfeatures at a time for memory efficiency
                for i in tqdm(range(0, self.num_features, dim_chunk)):
                    end_idx = min(i + dim_chunk, self.num_features)
                    inputs = torch.from_numpy(
                        get_col_slice(paper_feat, start_row_idx=0,
                                      end_row_idx=len(paper_feat),
                                      start_col_idx=i, end_col_idx=end_idx))
                    outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                    del inputs
                    save_col_slice(
                        x_from=outputs, x_to=x,
                        start_row_idx=dataset.num_papers,
                        end_row_idx=dataset.num_papers + dataset.num_authors,
                        start_col_idx=i, end_col_idx=end_idx)
                    del outputs

                edge_index = dataset.edge_index('author', 'institution')
                row, col = torch.from_numpy(edge_index)
                adj_t = SparseTensor(
                    row=col, col=row, sparse_sizes=(dataset.num_institutions,
                                                    dataset.num_authors),
                    is_sorted=False)

                print('Generating institution features...')
                # processing 64-dim subfeatures at a time for memory efficiency
                for i in tqdm(range(0, self.num_features, dim_chunk)):
                    end_idx = min(i + dim_chunk, self.num_features)
                    inputs = torch.from_numpy(
                        get_col_slice(
                            x, start_row_idx=dataset.num_papers,
                            end_row_idx=dataset.num_papers +
                            dataset.num_authors, start_col_idx=i,
                            end_col_idx=end_idx))
                    outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                    del inputs
                    save_col_slice(
                        x_from=outputs, x_to=x,
                        start_row_idx=dataset.num_papers + dataset.num_authors,
                        end_row_idx=N, start_col_idx=i, end_col_idx=end_idx)
                    del outputs

                x.flush()
                del x
                print(f'Done! [{time.perf_counter() - t:.2f}s]')

                with open(done_flag_path, 'w') as f:
                    f.write('done')

            except Exception:
                traceback.print_exc()
                if os.path.exists(path):
                    print(
                        'Removing unfinished full feat file due to exception')
                    os.remove(path)
                exit(-1)

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test'))
        self.test_idx.share_memory_()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        self.x = np.memmap(f'{dataset.root}/mag240m/full_feat.npy',
                           dtype=np.float16, mode='r',
                           shape=(N, self.num_features))
        self.y = torch.from_numpy(dataset.all_paper_label)

        path = f'{dataset.root}/mag240m/full_adj_t.pt'
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
        x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


class RGNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'rgat':
            self.convs.append(
                ModuleList([
                    GATConv(in_channels, hidden_channels // heads, heads,
                            add_self_loops=False) for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        GATConv(hidden_channels, hidden_channels // heads,
                                heads, add_self_loops=False)
                        for _ in range(num_relations)
                    ]))

        elif self.model == 'rgraphsage':
            self.convs.append(
                ModuleList([
                    SAGEConv(in_channels, hidden_channels, root_weight=False)
                    for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        SAGEConv(hidden_channels, hidden_channels,
                                 root_weight=False)
                        for _ in range(num_relations)
                    ]))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.skips.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skips.append(Linear(hidden_channels, hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]

            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                if subadj_t.nnz() > 0:
                    out += self.convs[i][j]((x, x_target), subadj_t)

            x = self.norms[i](out)
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        train_acc = self.acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        val_acc = self.acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return val_acc

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        test_acc = self.acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return test_acc

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
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes)

    if not args.evaluate:
        model = RGNN(args.model, datamodule.num_features,
                     datamodule.num_classes, args.hidden_channels,
                     datamodule.num_relations, num_layers=len(args.sizes),
                     dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
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
        model = RGNN.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()
        loader = datamodule.hidden_test_dataloader()

        model.eval()
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(int(args.device))
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}
        evaluator.save_test_submission(res, f'results/{args.model}')
