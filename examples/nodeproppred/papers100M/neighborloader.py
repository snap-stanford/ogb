import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                pbar.update(1)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all



def main():
    dataset = PygNodePropPredDataset(name='ogbn-papers100M', root="/data/ogb/")
    data = dataset[0]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12345)
    model = SAGE(data.num_features, 2, dataset.num_classes).to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    data.n_id = torch.arange(data.num_nodes)
    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=[10, 5],
        batch_size=1024,
        shuffle=False,
        num_workers=14,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 2):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            if hasattr(batch, 'adj_t'):
                edge_index = batch.adj_t
            else:
                edge_index = batch.edge_index
            out = model(batch.x, edge_index)
            batch_size = batch.batch_size
            out = out[:batch_size]
            target = batch.y[:batch_size]
            loss = F.cross_entropy(out, target.long().squeeze(1))
            loss.backward()
            optimizer.step()
    print("train finished")


if __name__ == "__main__":
    main()
