### example code of GIN using pytorch geometrics on protein function prediction dataset
### Warning: the dataset size is currently too large to be on GPU.
### (TODO) Implement Neighbor sampler variant.
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB

### for loading dataset
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
### for evaluation
from ogb.nodeproppred import Evaluator

criterion = torch.nn.BCEWithLogitsLoss()

class GINConv(MessagePassing):
    """
    - GIN architecture.
    - Assume both x and edge_attr have the dimensionality of emb_dim.
    """
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="mean")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        ### propagate = message -> aggr -> update
        h = (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.mlp(h)

        return out

    ### message to be aggregated
    ### x_j is the feature of source node
    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return aggr_out


class GIN(torch.nn.Module):
    def __init__(self, num_layer = 2, emb_dim = 50, num_task = 2):
        super(GIN, self).__init__()

        self.num_layer = num_layer

        self.gins = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layer):
            self.gins.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        self.node_encoder = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(8, emb_dim)

        self.node_pred_linear = torch.nn.Linear(emb_dim, num_task)

    def forward(self, data):
        h = self.node_encoder(data.x)
        edge_emb = self.edge_encoder(data.edge_attr)

        ### iterative message passing to obtain node embeddings
        for layer in range(self.num_layer):
            h = self.gins[layer](h, data.edge_index, edge_emb)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
        
        return self.node_pred_linear(h)


def train(model, data, train_idx, optimizer):
    model.train()

    pred = model(data)
    optimizer.zero_grad()
    loss = criterion(pred[train_idx].to(torch.float32), data.y[train_idx].to(torch.float32))
    loss.backward()
    optimizer.step()

    print(loss)

def eval(model, data, splitted_idx, evaluator):
    model.eval()
    y_pred = model(data).cpu().detach().numpy()
    y_true = data.y.cpu().detach().numpy()

    result = {}
    input_dict = {"y_true": y_true[splitted_idx["train"]], "y_pred": y_pred[splitted_idx["train"]]}
    result["train"] = evaluator.eval(input_dict)
    input_dict = {"y_true": y_true[splitted_idx["valid"]], "y_pred": y_pred[splitted_idx["valid"]]}
    result["valid"] = evaluator.eval(input_dict)
    input_dict = {"y_true": y_true[splitted_idx["test"]], "y_pred": y_pred[splitted_idx["test"]]}
    result["test"] = evaluator.eval(input_dict)

    return result


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GIN with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="proteinfunc",
                        help='dataset name (default: proteinfunc)')
    args = parser.parse_args()

    #device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu") ## only use cpu for now.

    ### automatic dataloading and splitting
    dataset = PygNodePropPredDataset(name = args.dataset)
    splitted_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    data = dataset[0]
    ### add dummy input node features
    data.x = torch.zeros(data.num_nodes, dtype = torch.long)
    data = data.to(device)
    model = GIN(num_task = dataset.num_tasks).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, args.epochs + 1):
        train(model, data, splitted_idx["train"], optimizer)
        print("Evaluating...")
        print(eval(model, data, splitted_idx, evaluator))


if __name__ == "__main__":
    main()
