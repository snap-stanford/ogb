import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch import Tensor
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=2):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, adjs: list) -> Tensor:
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                # x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x)

            x_all = torch.cat(xs, dim=0)

        return x_all


def main():
    dataset = PygNodePropPredDataset(name="ogbn-products", root="/data/ogb/")
    data = dataset[0]
    rank = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)
    torch.manual_seed(12345)
    model = SAGE(data.num_features, 256, dataset.num_classes).to(rank)
    y = data.y
    x = data.x
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    print("train_idx", train_idx.shape)
    data.n_id = torch.arange(data.num_nodes)
    print("data.n_id", data.n_id.shape)
    train_loader = NeighborSampler(
        data.edge_index,
        node_idx=train_idx,
        sizes=[5, 5],
        batch_size=1024,
        shuffle=False,
        num_workers=14,
    )
    print("loader finished")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochtimes = []
    for epoch in range(1, 2):
        model.train()
        for batch_size, n_id, adjs  in train_loader:
            optimizer.zero_grad()
            target_node = data.n_id[:batch_size]
            adjs = [adj.to(rank) for adj in adjs]
            out = model(x[n_id].to(rank), adjs)
            loss = F.nll_loss(out, y[target_node].squeeze(1).to(rank))
            loss.backward()
            optimizer.step()
        
    print("train finished")


if __name__ == "__main__":
    main()
