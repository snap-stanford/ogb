import copy
import argparse

import torch
import numpy as np
import torch.nn.functional as F

from torch_sparse import SparseTensor
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree, subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, concat=True))
        self.convs.append(SAGEConv(hidden_channels, out_channels, concat=True))

        for conv in self.convs:
            conv.aggr = 'add'

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)


class SAGEInference(torch.nn.Module):
    def __init__(self, weights):
        super(SAGEInference, self).__init__()
        self.weights = weights

    def forward(self, x, adj):
        out = x
        for i, (weight, bias) in enumerate(self.weights):
            tmp = adj @ out @ weight[weight.shape[0] // 2:]
            out = tmp + out @ weight[:weight.shape[0] // 2] + bias
            out = np.clip(out, 0, None) if i < len(self.weights) - 1 else out
        return out


def train(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_norm * data.edge_attr)
        loss = F.nll_loss(out, data.y.squeeze(1), reduction='none')
        loss = (loss * data.node_norm)[data.train_mask].sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def test(model, data, evaluator):
    print('Evaluating full-batch GNN on CPU...')

    weights = [(conv.weight.cpu().detach().numpy(),
                conv.bias.cpu().detach().numpy()) for conv in model.convs]
    model = SAGEInference(weights)

    x = data.x.numpy()
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
    adj = adj.sum(dim=1).pow(-1).view(-1, 1) * adj
    adj = adj.to_scipy(layout='csr')

    out = model(x, adj)

    y_true = data.y
    y_pred = torch.from_numpy(out).argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc


def to_inductive(data):
    mask = data.train_mask | data.valid_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.valid_mask = data.valid_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=5)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8000)
    parser.add_argument('--walk_length', type=int, default=3)
    parser.add_argument('--sample_coverage', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    # Create "inductive" subgraph containing only train and validation nodes.
    ind_data = to_inductive(copy.copy(data))
    row, col = ind_data.edge_index
    ind_data.edge_attr = 1. / degree(col, ind_data.num_nodes)[col]

    loader = GraphSAINTRandomWalkSampler(ind_data, batch_size=args.batch_size,
                                         walk_length=args.walk_length,
                                         num_steps=args.num_steps,
                                         sample_coverage=args.sample_coverage,
                                         save_dir=dataset.processed_dir,
                                         num_workers=args.num_workers)

    model = SAGE(ind_data.x.size(-1), args.hidden_channels, 47,
                 args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, loader, optimizer, device)
            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}')
        result = test(model, data, evaluator)
        logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
