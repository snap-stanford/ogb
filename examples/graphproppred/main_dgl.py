### example code of GIN using DGL
import torch
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB

### for loading dataset
from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset, collate_dgl
### for encoding raw molecule features
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
### for evaluation
from ogb.graphproppred import Evaluator

criterion = torch.nn.BCEWithLogitsLoss()

class GINConv(torch.nn.Module):
    """
    - GIN architecture.
    - Assume both node_feat and edge_feat have the dimensionality of emb_dim.
    """
    def __init__(self, emb_dim):
        super(GINConv, self).__init__()

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, graph, node_feat, edge_feat):
        graph = graph.local_var()
        graph.ndata['h_n'] = node_feat
        graph.edata['h_e'] = edge_feat

        ### u, v, e represent source nodes, destination nodes and edges among them
        graph.update_all(fn.u_add_e('h_n', 'h_e', 'm'), fn.sum('m', 'neigh'))
        rst = (1 + self.eps) * node_feat + graph.ndata['neigh']
        rst = self.mlp(rst)

        return rst


class GIN(torch.nn.Module):
    def __init__(self, num_layer = 5, emb_dim = 100, num_task = 2, device = "cpu"):
        super(GIN, self).__init__()

        self.num_layer = num_layer

        self.gins = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layer):
            self.gins.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        ### convenient module to encode/embed raw molecule node/edge features. (TODO) make it more efficient.
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        self.graph_pred_linear = torch.nn.Linear(emb_dim, num_task)
        self.device = device


    def forward(self, g):
        h_node = self.atom_encoder(g.ndata["feat"].to(self.device))
        h_edge = self.bond_encoder(g.edata["feat"].to(self.device))

         ### iterative message passing to obtain node embeddings
        for layer in range(self.num_layer):
            h_node = self.gins[layer](g, h_node, h_edge)
            h_node = self.batch_norms[layer](h_node)
            h_node = F.relu(h_node)

        ### pooling
        g.ndata['h_node'] = h_node
        h_graph = dgl.mean_nodes(g, 'h_node')
        
        return self.graph_pred_linear(h_graph)



def train(model, device, loader, optimizer):
    model.train()

    for step, (graphs, labels) in enumerate(tqdm(loader, desc="Iteration")):
        labels = labels.to(device)
        pred = model(graphs)
        optimizer.zero_grad()
        is_valid = labels == labels
        loss = criterion(pred.to(torch.float32)[is_valid], labels.to(torch.float32)[is_valid])
        loss.backward()
        optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (graphs, labels) in enumerate(tqdm(loader, desc="Iteration")):

        with torch.no_grad():
            pred = model(graphs)

        y_true.append(labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GIN with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-mol-tox21",
                        help='dataset name (default: ogbg-mol-tox21)')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


    ### automatic dataloading and splitting
    dataset = DglGraphPropPredDataset(name = args.dataset)
    splitted_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[splitted_idx["train"]], batch_size=args.batch_size, shuffle=True, collate_fn = collate_dgl, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[splitted_idx["valid"]], batch_size=args.batch_size, shuffle=False, collate_fn = collate_dgl, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[splitted_idx["test"]], batch_size=args.batch_size, shuffle=False, collate_fn = collate_dgl, num_workers = args.num_workers)

    model = GIN(num_task = dataset.num_tasks, device = device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer)
        #print("Evaluating training...")
        #print(eval(model, device, train_loader, evaluator))
        print("Evaluating validation:")
        print(eval(model, device, valid_loader, evaluator))


if __name__ == "__main__":
    main()