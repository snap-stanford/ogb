import argparse

import torch
from torch_sparse import transpose
from torch_geometric.nn import MetaPath2Vec

from ogb.nodeproppred import PygNodePropPredDataset


@torch.no_grad()
def save_embedding(model):
    embedding = model('paper').cpu()
    torch.save(embedding, 'embedding.pt')


def main():
    parser = argparse.ArgumentParser(description='OGBN-MAG (MetaPath2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=64)
    parser.add_argument('--context_size', type=int, default=7)
    parser.add_argument('--walks_per_node', type=int, default=5)
    parser.add_argument('--num_negative_samples', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=100)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset('ogbn-mag')
    data = dataset[0]

    # We need to add reverse edges to the heterogeneous graph.
    data.edge_index_dict[('institution', 'employs', 'author')] = transpose(
        data.edge_index_dict[('author', 'affiliated_with', 'institution')],
        None, m=data.num_nodes_dict['author'],
        n=data.num_nodes_dict['institution'])[0]
    data.edge_index_dict[('paper', 'written_by', 'author')] = transpose(
        data.edge_index_dict[('author', 'writes', 'paper')], None,
        m=data.num_nodes_dict['author'], n=data.num_nodes_dict['paper'])[0]
    data.edge_index_dict[('field_of_study', 'contains', 'paper')] = transpose(
        data.edge_index_dict[('paper', 'has_topic', 'field_of_study')], None,
        m=data.num_nodes_dict['paper'],
        n=data.num_nodes_dict['field_of_study'])[0]
    print(data)

    metapath = [
        ('author', 'writes', 'paper'),
        ('paper', 'has_topic', 'field_of_study'),
        ('field_of_study', 'contains', 'paper'),
        ('paper', 'written_by', 'author'),
        ('author', 'affiliated_with', 'institution'),
        ('institution', 'employs', 'author'),
        ('author', 'writes', 'paper'),
        ('paper', 'cites', 'paper'),
        ('paper', 'written_by', 'author'),
    ]

    model = MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
                         metapath=metapath, walk_length=64, context_size=7,
                         walks_per_node=5, num_negative_samples=5,
                         sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 1000 == 0:  # Save model every 1000 steps.
                save_embedding(model)
        save_embedding(model)


if __name__ == "__main__":
    main()
