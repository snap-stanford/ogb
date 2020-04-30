import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.transforms import SIGN

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers + 1):
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.lin = torch.nn.Linear((num_layers + 1) * hidden_channels,
                                   out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, xs):
        outs = []
        for x, lin in zip(xs, self.lins):
            out = F.dropout(F.relu(lin(x)), p=0.5, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.lin(x)
        return torch.log_softmax(x, dim=-1)


def train(model, xs, y_true, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(xs)
    loss = F.nll_loss(out, y_true.squeeze(1))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, xs, y_true, evaluator):
    model.eval()

    y_preds = []
    loader = DataLoader(range(y_true.size(0)), batch_size=400000)
    for perm in loader:
        y_pred = model([x[perm] for x in xs]).argmax(dim=-1, keepdim=True)
        y_preds.append(y_pred.cpu())
    y_pred = torch.cat(y_preds, dim=0)

    return evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })['acc']


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (SIGN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = SIGN(args.num_layers)(dataset[0])  # This might take a while.

    xs = [data.x] + [data[f'x{i}'] for i in range(1, args.num_layers + 1)]
    xs_train = [x[split_idx['train']].to(device) for x in xs]
    xs_valid = [x[split_idx['valid']].to(device) for x in xs]
    xs_test = [x[split_idx['test']].to(device) for x in xs]

    y_train_true = data.y[split_idx['train']].to(device)
    y_valid_true = data.y[split_idx['valid']].to(device)
    y_test_true = data.y[split_idx['test']].to(device)

    model = MLP(data.x.size(-1), args.hidden_channels, dataset.num_classes, args.num_layers,
                args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, xs_train, y_train_true, optimizer)

            train_acc = test(model, xs_train, y_train_true, evaluator)
            valid_acc = test(model, xs_valid, y_valid_true, evaluator)
            test_acc = test(model, xs_test, y_test_true, evaluator)
            result = (train_acc, valid_acc, test_acc)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
