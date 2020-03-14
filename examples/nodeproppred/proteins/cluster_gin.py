import argparse

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN

from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import MessagePassing

from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator

from logger import Logger


class GINConv(MessagePassing):
    def __init__(self, hidden_channels):
        super(GINConv, self).__init__(aggr='mean')

        self.mlp = Seq(Lin(hidden_channels, 2 * hidden_channels), ReLU(),
                       Lin(2 * hidden_channels, hidden_channels))
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))

    def reset_parameters(self):
        self.mlp[0].reset_parameters()
        self.mlp[2].reset_parameters()
        self.eps.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        h = (1 + self.eps) * x + self.propagate(edge_index, x=x,
                                                edge_attr=edge_attr)
        return self.mlp(h)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class GIN(torch.nn.Module):
    def __init__(self, in_node_channels, in_edge_channels, hidden_channels,
                 out_channels, num_layers, dropout):
        super(GIN, self).__init__()

        self.node_encoder = Lin(in_node_channels, hidden_channels)
        self.edge_encoder = Lin(in_edge_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINConv(hidden_channels))

        self.lin = Lin(hidden_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x


def train(model, loader, optimizer, device):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)[data.train_mask]
        loss = criterion(out, data.y[data.train_mask].to(torch.float))
        loss.backward()
        optimizer.step()

        num_examples = data.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, loader, evaluator, device):
    model.eval()

    train_masks, valid_masks, test_masks = [], [], []
    y_trues, y_preds = [], []

    for data in loader:
        y_trues.append(data.y.clone())
        train_masks.append(data.train_mask.clone())
        valid_masks.append(data.valid_mask.clone())
        test_masks.append(data.test_mask.clone())

        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        y_preds.append(out.cpu())

    train_mask = torch.cat(train_masks, dim=0)
    valid_mask = torch.cat(valid_masks, dim=0)
    test_mask = torch.cat(test_masks, dim=0)
    y_true = torch.cat(y_trues, dim=0)
    y_pred = torch.cat(y_preds, dim=0)

    train_rocauc = evaluator.eval({
        'y_true': y_true[train_mask],
        'y_pred': y_pred[train_mask],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': y_true[valid_mask],
        'y_pred': y_pred[valid_mask],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': y_true[test_mask],
        'y_pred': y_pred[test_mask],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_features', action='store_true')
    parser.add_argument('--num_partitions', type=int, default=700)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins')
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in splitted_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)

    if not args.use_node_features:
        cluster_data.data.x = torch.ones(cluster_data.data.num_nodes, 1)
    else:
        cluster_data.data.x = cluster_data.data.x.to(torch.float)

    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    model = GIN(cluster_data.data.x.size(-1), data.edge_attr.size(-1),
                args.hidden_channels, 112, args.num_layers,
                args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, loader, optimizer, device)

            if epoch % args.eval_steps == 0:
                result = test(model, loader, evaluator, device)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
