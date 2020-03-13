import argparse

import torch
import torch.nn.functional as F

from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator

from logger import Logger


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return torch.log_softmax(x, dim=-1)


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

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return torch.log_softmax(x, dim=-1)


def train(model, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[data.train_mask]
        loss = F.nll_loss(out, data.y.squeeze(1)[data.train_mask])
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
        out = model(data.x, data.edge_index)
        y_preds.append(out.argmax(dim=-1, keepdim=True).cpu())

    train_mask = torch.cat(train_masks, dim=0)
    valid_mask = torch.cat(valid_masks, dim=0)
    test_mask = torch.cat(test_masks, dim=0)
    y_true = torch.cat(y_trues, dim=0)
    y_pred = torch.cat(y_preds, dim=0)

    train_acc = evaluator.eval({
        'y_true': y_true[train_mask],
        'y_pred': y_pred[train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[valid_mask],
        'y_pred': y_pred[valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[test_mask],
        'y_pred': y_pred[test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_partitions', type=int, default=15000)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products')
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in splitted_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)

    loader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    if args.use_sage:
        model = SAGE(data.x.size(-1), args.hidden_channels, 47,
                     args.num_layers, args.dropout).to(device)
    else:
        model = GCN(data.x.size(-1), args.hidden_channels, 47, args.num_layers,
                    args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, loader, optimizer, device)
            result = test(model, loader, evaluator, device)
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
