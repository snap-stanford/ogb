import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_mean

from ogb.linkproppred.dataset_pyg import PygLinkPropPredDataset
from ogb.linkproppred import Evaluator

from logger import Logger


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x_i, x_j, y=None):
        x = x_i * x_j
        if y is not None:
            x = torch.cat([x, y], dim=-1)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x.squeeze()


def train(predictor, product_emb, user_emb, product_x, splitted_edge, optimizer, batch_size):
    predictor.train()

    train_edge = splitted_edge['train_edge']
    train_label = splitted_edge['train_edge_label']

    total_loss = total_examples = 0
    for i, perm in enumerate(DataLoader(range(train_edge.size(1)), batch_size, shuffle=True)):
        optimizer.zero_grad()
        if product_x is not None:
            product_h = product_x[train_edge[0][perm]]
        else:
            product_h = None
        out = predictor(product_emb[train_edge[0][perm]], user_emb[train_edge[1][perm]], product_h)
        loss = F.mse_loss(out, train_label[perm].to(out.device))
        loss.backward()
        optimizer.step()

        num_examples = out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        if i > 4:
            continue

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, product_emb, user_emb, product_x, splitted_edge, evaluator, batch_size):
    predictor.eval()

    train_y_preds = []
    train_edge = splitted_edge['train_edge']
    for perm in DataLoader(range(train_edge.size(1)), batch_size):
        if product_x is not None:
            product_h = product_x[train_edge[0][perm]]
        else:
            product_h = None
        pred = predictor(product_emb[train_edge[0][perm]], user_emb[train_edge[1][perm]], product_h)
        train_y_preds += [pred.clamp_(1, 5).cpu()]
    
    valid_y_preds = []
    valid_edge = splitted_edge['valid_edge']
    for perm in DataLoader(range(valid_edge.size(1)), batch_size):
        if product_x is not None:
            product_h = product_x[valid_edge[0][perm]]
        else:
            product_h = None
        pred = predictor(product_emb[valid_edge[0][perm]], user_emb[valid_edge[1][perm]], product_h)
        valid_y_preds += [pred.clamp_(1, 5).cpu()]

    test_y_preds = []
    test_edge = splitted_edge['test_edge']
    for perm in DataLoader(range(test_edge.size(1)), batch_size):
        if product_x is not None:
            product_h = product_x[test_edge[0][perm]]
        else:
            product_h = None
        pred = predictor(product_emb[test_edge[0][perm]], user_emb[test_edge[1][perm]], product_h)
        test_y_preds += [pred.clamp_(1, 5).cpu()]

    train_y_pred = torch.cat(train_y_preds, dim=0)
    valid_y_pred = torch.cat(valid_y_preds, dim=0)
    test_y_pred = torch.cat(test_y_preds, dim=0)

    train_rmse = evaluator.eval({
        'y_true': splitted_edge['train_edge_label'],
         'y_pred': train_y_pred,
    })['rmse']
    valid_rmse = evaluator.eval({
        'y_true': splitted_edge['valid_edge_label'],
        'y_pred': valid_y_pred,
    })['rmse']
    test_rmse = evaluator.eval({
        'y_true': splitted_edge['test_edge_label'],
        'y_pred': test_y_pred,
    })['rmse']

    return train_rmse, valid_rmse, test_rmse


def main():
    parser = argparse.ArgumentParser(description='OGBL-Reviews (MLP)')
    parser.add_argument('--suffix', type=str, default='groc')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--use_node_features', action='store_true')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=f'ogbl-reviews-{args.suffix}')
    splitted_edge = dataset.get_edge_split()
    data = dataset[0]

    product_idx = (data.x[:, 0] == 0).nonzero().flatten()
    user_idx = (data.x[:, 0] == 1).nonzero().flatten()

    user_emb = torch.nn.Parameter(
        torch.Tensor(user_idx.size(0), args.hidden_channels).to(device))

    if args.use_node_features:
        product_x = data.x[product_idx, 1:].to(device)
    else:
        product_x = None

    product_emb = torch.nn.Parameter(
        torch.Tensor(product_idx.size(0), args.hidden_channels).to(device))

    for split in ['train', 'valid', 'test']:
        edge = splitted_edge[f'{split}_edge'].t()
        edge[1] -= product_emb.size(0)
        splitted_edge[f'{split}_edge'] = edge

    train_products = splitted_edge['train_edge'][1]
    all_products = torch.cat([splitted_edge['train_edge'][1], splitted_edge['valid_edge'][1], splitted_edge['test_edge'][1]], 0)


    predictor = LinkPredictor(
        args.hidden_channels + (300 if product_x is not None else 0), args.hidden_channels, 1,
        args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name=f'ogbl-reviews-{args.suffix}')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        predictor.reset_parameters()
        glorot(user_emb)
        glorot(product_emb)

        optimizer = torch.optim.Adam([user_emb, product_emb] + list(predictor.parameters()),
                                     lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(predictor, product_emb, user_emb, product_x, splitted_edge,
                         optimizer, args.batch_size)
            result = test(predictor, product_emb, user_emb, product_x, splitted_edge,
                          evaluator, args.batch_size)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_rmse, valid_rmse, test_rmse = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {train_rmse:.4f}, '
                      f'Valid: {valid_rmse:.4f}, '
                      f'Test: {test_rmse:.4f}')

        logger.print_statistics(run)
    logger.print_statistics()

    if args.out_file is not None:
        logger.save(args.out_file)


if __name__ == "__main__":
    main()
