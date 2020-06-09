import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ogb.nodeproppred import Evaluator

from logger import Logger


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def train(model, device, train_loader, optimizer):
    model.train()

    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, device, loader, evaluator):
    model.eval()

    y_pred, y_true = [], []
    for x, y in tqdm(loader):
        x = x.to(device)
        out = model(x)

        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_true.append(y)

    return evaluator.eval({
        "y_true": torch.cat(y_true, dim=0),
        "y_pred": torch.cat(y_pred, dim=0),
    })['acc']


def main():
    parser = argparse.ArgumentParser(description='OGBN-papers100M (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--use_sgc_embedding', action='store_true')
    parser.add_argument('--num_sgc_iterations', type=int, default = 3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)



    if not args.use_sgc_embedding:
        try:
            data_dict = torch.load('data_dict.pt')
        except:
            raise RuntimeError('data_dict.pt not found. Need to run python node2vec.py first')
        x = data_dict['node_feat']
        split_idx = data_dict['split_idx']
        y = data_dict['label'].to(torch.long)

        if args.use_node_embedding:
            x = torch.cat([x, data_dict['node2vec_embedding']], dim=-1)

        print(x.shape)
        
    else:
        if args.use_node_embedding:
            raise ValueError('No option to use node embedding and sgc embedding at the same time.')
        else:
            try:
                sgc_dict = torch.load('sgc_dict.pt')
            except:
                raise RuntimeError('sgc_dict.pt not found. Need to run python sgc.py first')

            x = sgc_dict['sgc_embedding'][args.num_sgc_iterations]
            split_idx = sgc_dict['split_idx']
            y = sgc_dict['label'].to(torch.long)

    train_dataset = SimpleDataset(x[split_idx['train']], y[split_idx['train']])
    valid_dataset = SimpleDataset(x[split_idx['valid']], y[split_idx['valid']])
    test_dataset = SimpleDataset(x[split_idx['test']], y[split_idx['test']])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = MLP(x.size(-1), args.hidden_channels, 172, args.num_layers,
                args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-papers100M')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            train(model, device, train_loader, optimizer)
            train_acc = test(model, device, train_loader, evaluator)
            valid_acc = test(model, device, valid_loader, evaluator)
            test_acc = test(model, device, test_loader, evaluator)

            logger.add_result(run, (train_acc, valid_acc, test_acc))

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
