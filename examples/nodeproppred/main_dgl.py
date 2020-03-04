import argparse
import dgl.function as fn
import torch
import torch.nn as nn

from ogb.nodeproppred import Evaluator
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from torch.optim import Adam

class GINConv(nn.Module):
    def __init__(self, in_dim, has_edge_feats):
        super(GINConv, self).__init__()

        self.has_edge_feats = has_edge_feats
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * in_dim),
            nn.BatchNorm1d(2 * in_dim),
            nn.ReLU(),
            nn.Linear(2 * in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU()
        )

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        if self.has_edge_feats:
            g.edata['he'] = edge_feats
            g.update_all(fn.u_add_e('hv', 'he', 'm'), fn.mean('m', 'hv_new'))
        else:
            g.update_all(fn.copy_src('hv', 'm'), fn.mean('m', 'hv_new'))
        node_feats = (1 + self.eps) * g.ndata['hv'] + g.ndata['hv_new']

        return self.mlp(node_feats)

class GIN(nn.Module):
    def __init__(self, has_node_feats, has_edge_feats,
                 in_dim=8, num_layer=2, emb_dim=50, num_task=112):
        super(GIN, self).__init__()

        self.has_node_feats = has_node_feats
        self.has_edge_feats = has_edge_feats
        if has_node_feats:
            self.node_encoder = nn.Linear(in_dim, emb_dim)
        else:
            self.node_encoder = nn.Embedding(1, emb_dim)
        if has_edge_feats:
            self.edge_encoder = nn.Linear(in_dim, emb_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layer):
            self.gnn_layers.append(GINConv(emb_dim, has_edge_feats))

        self.pred_out = nn.Linear(emb_dim, num_task)

    @property
    def device(self):
        return self.pred_out.weight.device

    def forward(self, g, node_feats, edge_feats):
        assert self.has_node_feats == (node_feats is not None), 'Edge features required'
        assert self.has_edge_feats == (edge_feats is not None), 'Node features required'

        if node_feats is None:
            node_types = torch.zeros(g.number_of_nodes()).long().to(self.device)
            node_feats = self.node_encoder(node_types)
        else:
            node_feats = self.node_encoder(node_feats)

        if edge_feats is not None:
            edge_feats = self.edge_encoder(edge_feats)

        # Message passing
        for gnn_layer in self.gnn_layers:
            node_feats = gnn_layer(g, node_feats, edge_feats)

        return self.pred_out(node_feats)

def prepare_feats(model, graph):
    if model.has_node_feats:
        node_feats = graph.ndata['feat']
    else:
        node_feats = None
    if model.has_edge_feats:
        edge_feats = graph.edata['feat']
    else:
        edge_feats = None
    return node_feats, edge_feats

def train(model, graph, train_node_idx, criterion, optimizer):
    model.train()
    node_feats, edge_feats = prepare_feats(model, graph)
    logits = model(graph, node_feats, edge_feats)[train_node_idx]

    labels = graph.ndata['labels'][train_node_idx]
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.cpu().detach().data.item()

def eval(model, graph, splitted_idx, evaluator):
    model.eval()
    node_feats, edge_feats = prepare_feats(model, graph)
    with torch.no_grad():
        logits = model(graph, node_feats, edge_feats)

    labels = graph.ndata['labels'].cpu().numpy()
    logits = logits.detach().cpu().numpy()

    train_score = evaluator.eval({
        "y_true": labels[splitted_idx["train"]],
        "y_pred": logits[splitted_idx["train"]]
    })
    val_score = evaluator.eval({
        "y_true": labels[splitted_idx["valid"]],
        "y_pred": logits[splitted_idx["valid"]]
    })
    test_score = evaluator.eval({
        "y_true": labels[splitted_idx["test"]],
        "y_pred": logits[splitted_idx["test"]]
    })

    return train_score['rocauc'], val_score['rocauc'], test_score['rocauc']

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GIN with DGL')
    parser.add_argument('-n', '--node-feats', action='store_true',
                        help='Whether to use node features for modeling')
    parser.add_argument('-e', '--edge-feats', action='store_true',
                        help='Whether to use edge features for modeling')
    parser.add_argument('-nl', '--num-layers', type=int, default=2,
                        help='Number of GIN layers to use (default: 2)')
    parser.add_argument('-ed', '--embed-dim', type=int, default=50,
                        help='Hidden size in GIN (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    args = parser.parse_args()
    args.dataset = 'ogbn-proteins'

    assert args.node_feats or args.edge_feats, \
        'At least one of the node/edge features is required for modeling.'

    # Only use CPU for now
    device = torch.device("cpu")

    # Data loading and splitting
    dataset = DglNodePropPredDataset(name=args.dataset)
    print(dataset.meta_info[args.dataset])
    splitted_idx = dataset.get_idx_split()

    # Change the dtype and device of tensors
    graph = dataset.graph[0]
    graph.ndata['labels'] = dataset.labels.float().to(args.device)
    if args.node_feats:
        graph.ndata['feat'] = graph.ndata['feat'].float().to(args.device)
    if args.edge_feats:
        graph.edata['feat'] = graph.edata['feat'].float().to(args.device)

    model = GIN(has_node_feats=args.node_feats,
                has_edge_feats=args.edge_feats,
                num_layer=args.num_layers,
                emb_dim=args.embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    evaluator = Evaluator(args.dataset)

    for epoch in range(args.epochs):
        loss = train(model, graph, splitted_idx['train'], criterion, optimizer)
        train_score, val_score, test_score = eval(model, graph, splitted_idx, evaluator)
        print('Epoch {:d}/{:d} | loss {:.4f} | train rocauc {:.4f} | '
              'val rocauc {:.4f} | test rocauc {:.4f}'.format(
            epoch + 1, args.epochs, loss, train_score, val_score, test_score))

if __name__ == '__main__':
    main()
