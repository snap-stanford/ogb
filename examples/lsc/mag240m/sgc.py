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
                 batch_norm: bool = True, relu_last: bool = False):
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
        self.relu_last = relu_last

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        for lin, batch_norm in zip(self.lins[:-1], self.batch_norms):
            x = lin(x)
            if self.relu_last:
                x = batch_norm(x).relu_()
            else:
                x = batch_norm(x.relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def train(model, x_train, y_train, batch_size, optimizer):
    model.train()

    total_loss = 0
    for idx in DataLoader(range(y_train.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()

    return total_loss / y_train.size(0)


@torch.no_grad()
def test(model, x_eval, y_eval, evaluator):
    model.eval()
    y_pred = model(x_eval).argmax(dim=-1)
    return evaluator.eval({'y_true': y_eval, 'y_pred': y_pred})['acc']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3),
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_last', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=380000)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test-dev')

    t = time.perf_counter()
    print('Reading node features...', end=' ', flush=True)
    x_train = np.load(f'{dataset.dir}/x_train_{args.layer}.npy')
    x_train = torch.from_numpy(x_train).to(device)
    x_valid = np.load(f'{dataset.dir}/x_valid_{args.layer}.npy')
    x_valid = torch.from_numpy(x_valid).to(device)
    x_test = np.load(f'{dataset.dir}/x_test_{args.layer}.npy')
    x_test = torch.from_numpy(x_test).to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    y_train = torch.from_numpy(dataset.paper_label[train_idx])
    y_train = y_train.to(device, torch.long)
    y_valid = torch.from_numpy(dataset.paper_label[valid_idx])
    y_valid = y_valid.to(device, torch.long)

    model = MLP(dataset.num_paper_features, args.hidden_channels,
                dataset.num_classes, args.num_layers, args.dropout,
                not args.no_batch_norm, args.relu_last).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f'#Params: {num_params}')

    best_valid_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, x_train, y_train, args.batch_size, optimizer)
        train_acc = test(model, x_train, y_train, evaluator)
        valid_acc = test(model, x_valid, y_valid, evaluator)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            with torch.no_grad():
                model.eval()
                res = {'y_pred': model(x_test).argmax(dim=-1)}
                evaluator.save_test_submission(res, 'results/sgc', mode = 'test-dev')
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                  f'Best: {best_valid_acc:.4f}')
