import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj

from ogb.nodeproppred import PygNodePropPredDataset


def main():
    parser = argparse.ArgumentParser(description='OGBN-papers100M (MLP)')
    parser.add_argument('--num_propagations', type=int, default=3)
    parser.add_argument('--dropedge_rate', type=float, default=0.4)
    args = parser.parse_args()

    # SGC pre-processing ######################################################

    dataset = PygNodePropPredDataset('ogbn-papers100M')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    x = data.x.numpy()
    N = data.num_nodes

    print('Making the graph undirected.')
    ### Randomly drop some edges to save computation
    data.edge_index, _ = dropout_adj(data.edge_index, p = args.dropedge_rate, num_nodes= data.num_nodes)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    print(data)

    row, col = data.edge_index

    print('Computing adj...')

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    adj = adj.to_scipy(layout='csr')


    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_idx = torch.cat([train_idx, valid_idx, test_idx])
    mapped_train_idx = torch.arange(len(train_idx))
    mapped_valid_idx = torch.arange(len(train_idx), len(train_idx) + len(valid_idx))
    mapped_test_idx = torch.arange(len(train_idx) + len(valid_idx), len(train_idx) + len(valid_idx) + len(test_idx))

    sgc_dict = {}
    sgc_dict['label'] = data.y.data[all_idx].to(torch.long)
    sgc_dict['split_idx'] = {'train': mapped_train_idx, 'valid': mapped_valid_idx, 'test': mapped_test_idx}


    sgc_dict['sgc_embedding'] = []
    sgc_dict['sgc_embedding'].append(torch.from_numpy(x[all_idx]).to(torch.float))

    print('Start SGC processing')

    for _ in tqdm(range(args.num_propagations)):
        x = adj @ x
        sgc_dict['sgc_embedding'].append(torch.from_numpy(x[all_idx]).to(torch.float))


    print(sgc_dict)

    torch.save(sgc_dict, 'sgc_dict.pt')


if __name__ == "__main__":
    main()
