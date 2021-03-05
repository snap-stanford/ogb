import argparse

import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected, dropout_adj
from tqdm import tqdm

from ogb.nodeproppred import PygNodePropPredDataset

# save both node2vec embeddings and raw node features, labels, and split.
# Only save nodes that are labeled (specified by save_idx).
def save_data_dict(model, data, split_idx, save_file):
    data_dict = {}
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, valid_idx, test_idx])
    mapped_train_idx = torch.arange(len(train_idx))
    mapped_valid_idx = torch.arange(len(train_idx), len(train_idx) + len(valid_idx))
    mapped_test_idx = torch.arange(len(train_idx) + len(valid_idx), len(train_idx) + len(valid_idx) + len(test_idx))

    data_dict['node2vec_embedding'] = model.embedding.weight.data[all_idx].cpu()
    data_dict['node_feat'] = data.x.data[all_idx]
    data_dict['label'] = data.y.data[all_idx].to(torch.long)

    data_dict['split_idx'] = {'train': mapped_train_idx, 'valid': mapped_valid_idx, 'test': mapped_test_idx}

    print(data_dict)

    torch.save(data_dict, save_file)


def main():
    parser = argparse.ArgumentParser(description='OGBN-Papers100M (Node2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=20)
    parser.add_argument('--context_size', type=int, default=10)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--dropedge_rate', type=float, default=0.4)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-papers100M')

    split_idx = dataset.get_idx_split()

    data = dataset[0]

    # if args.add_inverse:

    print('Making the graph undirected.')
    ### Randomly drop some edges to avoid segmentation fault
    data.edge_index, _ = dropout_adj(data.edge_index, p = args.dropedge_rate, num_nodes= data.num_nodes)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    filename = 'data_dict.pt'

    print(data)

    model = Node2Vec(data.edge_index, args.embedding_dim, args.walk_length,
                     args.context_size, args.walks_per_node,
                     sparse=True).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True,
                          num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    print('Saving data_dict before training...')
    save_data_dict(model, data, split_idx, save_file = filename)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in tqdm(enumerate(loader)):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 1000 == 0:  # Save model every 1000 steps.
                print('Saving data dict...')
                save_data_dict(model, data, split_idx, save_file = filename)

        print('Saving data dict...')
        save_data_dict(model, data, split_idx, save_file = filename)


if __name__ == "__main__":
    main()
