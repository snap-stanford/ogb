import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F

from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.nn.inits import glorot, zeros

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj):
        return adj @ x @ self.weight


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x.log_softmax(dim=-1)


class SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.root_weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.root_weight)
        zeros(self.bias)

    def forward(self, x, adj):
        out = adj.matmul(x, reduce="mean") @ self.weight
        out = out + x @ self.root_weight + self.bias
        return out


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x.log_softmax(dim=-1)


def train(model, x, adj, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x, adj)[train_idx]
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, adj, y_true, split_idx, evaluator):
    model.eval()

    out = model(x, adj)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()

    data = dataset[0]

    x = data.x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    edge_index = data.edge_index.to(device)
    edge_index = to_undirected(edge_index, data.num_nodes)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])

    if args.use_sage:
        model = SAGE(data.x.size(-1), args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                    args.num_layers, args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj = adj.set_diag()
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, adj, y_true, train_idx, optimizer)
            result = test(model, x, adj, y_true, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
