import argparse

import torch
from torch.utils.data import DataLoader

from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected

from ogb.nodeproppred import PygNodePropPredDataset


@torch.no_grad()
def save_embedding(model):
    model.eval()
    device = model.embedding.weight.device
    embedding = model(torch.arange(model.num_nodes, device=device)).cpu()
    torch.save(embedding, 'embedding.pt')


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (Node2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    edge_index = data.edge_index.to(device)
    edge_index = to_undirected(edge_index, data.num_nodes)

    model = Node2Vec(data.num_nodes, args.embedding_dim, args.walk_length,
                     args.context_size, args.walks_per_node).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(torch.arange(data.num_nodes),
                        batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, subset in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(edge_index, subset.to(edge_index.device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 100 == 0:  # Save model every 100 steps.
                save_embedding(model)
        save_embedding(model)


if __name__ == "__main__":
    main()
