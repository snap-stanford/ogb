# NOTE: More than 256GB CPU memory required to run this script.
#       Use `--low-memory` to reduce memory consumption by using half-precision

import os.path as osp
import time
import argparse

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import LabelPropagation
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=3),
    parser.add_argument('--alpha', type=float, default=0.9),
    parser.add_argument('--low-memory', action='store_true'),
    args = parser.parse_args()
    print(args)

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    t = time.perf_counter()
    print('Reading adjacency matrix...', end=' ', flush=True)
    path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
    if osp.exists(path):
        adj_t = torch.load(path)
    else:
        edge_index = dataset.edge_index('paper', 'cites', 'paper')
        edge_index = torch.from_numpy(edge_index)
        adj_t = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            sparse_sizes=(dataset.num_papers, dataset.num_papers),
            is_sorted=True)
        adj_t = adj_t.to_symmetric()
        torch.save(adj_t, path)
    adj_t = gcn_norm(adj_t, add_self_loops=False)
    if args.low_memory:
        adj_t = adj_t.to(torch.half)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test-dev')

    y_train = torch.from_numpy(dataset.paper_label[train_idx]).to(torch.long)
    y_valid = torch.from_numpy(dataset.paper_label[valid_idx]).to(torch.long)

    model = LabelPropagation(args.num_layers, args.alpha)

    N, C = dataset.num_papers, dataset.num_classes

    t = time.perf_counter()
    print('Propagating labels...', end=' ', flush=True)
    if args.low_memory:
        y = torch.zeros(N, C, dtype=torch.half)
        y[train_idx] = F.one_hot(y_train, C).to(torch.half)
        out = model(y, adj_t, post_step=lambda x: x)
        y_pred = out.argmax(dim=-1)
    else:
        y = torch.zeros(N, C)
        y[train_idx] = F.one_hot(y_train, C).to(torch.float)
        out = model(y, adj_t)
        y_pred = out.argmax(dim=-1)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    train_acc = evaluator.eval({
        'y_true': y_train,
        'y_pred': y_pred[train_idx]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_valid,
        'y_pred': y_pred[valid_idx]
    })['acc']
    print(f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}')

    res = {'y_pred': y_pred[test_idx]}
    evaluator.save_test_submission(res, 'results/label_prop', mode = 'test-dev')
