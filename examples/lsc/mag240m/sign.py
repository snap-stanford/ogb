import time
import argparse

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import ModuleList, Linear, BatchNorm1d, Identity

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT


class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = True, relu_first: bool = False):
        super(MLP, self).__init__()

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.batch_norms = ModuleList()
        for _ in range(num_layers - 1):
            norm = BatchNorm1d(hidden_channels) if batch_norm else Identity()
            self.batch_norms.append(norm)

        self.dropout = dropout
        self.relu_first = relu_first

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        for lin, batch_norm in zip(self.lins[:-1], self.batch_norms):
            x = lin(x)
            if self.relu_first:
                x = batch_norm(x.relu_())
            else:
                x = batch_norm(x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class SIGN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_embeddings: int, num_layers: int,
                 dropout: float = 0.0, batch_norm: bool = True,
                 relu_first: bool = False):
        super(SIGN, self).__init__()

        self.mlps = ModuleList()
        for _ in range(num_embeddings):
            mlp = MLP(in_channels, hidden_channels, hidden_channels,
                      num_layers, dropout, batch_norm, relu_first)
            self.mlps.append(mlp)

        self.mlp = MLP(num_embeddings * hidden_channels, hidden_channels,
                       out_channels, num_layers, dropout, batch_norm,
                       relu_first)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()
        self.reset_parameters()

    def forward(self, xs):
        out = []
        for x, mlp in zip(xs, self.mlps):
            out.append(mlp(x))
        out = torch.cat(out, dim=-1).relu_()
        return self.mlp(out)


def train(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    for xs, y in loader:
        xs = [x.to(device) for x in xs]
        y = y.to(torch.long).to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(xs), y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader, evaluator, device):
    model.eval()

    y_true, y_pred = [], []
    for xs, y in loader:
        xs = [x.to(device) for x in xs]
        y_true.append(y.to(torch.long))
        y_pred.append(model(xs).argmax(dim=-1).cpu())

    return evaluator.eval({
        'y_true': torch.cat(y_true, dim=0),
        'y_pred': torch.cat(y_pred, dim=0)
    })['acc']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_hops', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2),
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_first', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, split):
            self.xs = []
            idx = dataset.get_idx_split(split)

            t = time.perf_counter()
            print(f'Reading {split} node features...', end=' ', flush=True)
            x = dataset.paper_feat[idx]
            self.xs.append(torch.from_numpy(x).to(torch.float))
            for i in range(1, args.num_hops + 1):
                x = np.load(f'{dataset.dir}/x_{split}_{i}.npy')
                self.xs.append(torch.from_numpy(x))
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            self.y = torch.from_numpy(dataset.paper_label[idx])

        def __len__(self):
            return self.xs[0].size(0)

        def __getitem__(self, idx):
            return [x[idx] for x in self.xs], self.y[idx]

    train_dataset = MyDataset(split='train')
    valid_dataset = MyDataset(split='valid')
    test_dataset = MyDataset(split='test-dev')

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              num_workers=6, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size)
    test_loader = DataLoader(test_dataset, args.batch_size)

    model = SIGN(dataset.num_paper_features, args.hidden_channels,
                 dataset.num_classes, args.num_hops + 1, args.num_layers,
                 args.dropout, not args.no_batch_norm,
                 args.relu_first).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f'#Params: {num_params}')

    best_valid_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, optimizer, device)
        train_acc = test(model, train_loader, evaluator, device)
        valid_acc = test(model, valid_loader, evaluator, device)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            with torch.no_grad():
                model.eval()
                y_pred = []
                for xs, _ in test_loader:
                    xs = [x.to(device) for x in xs]
                    y_pred.append(model(xs).argmax(dim=-1).cpu())
                res = {'y_pred': torch.cat(y_pred, dim=0)}
                evaluator.save_test_submission(res, 'results/sign', mode = 'test-dev')
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                  f'Best: {best_valid_acc:.4f}')
