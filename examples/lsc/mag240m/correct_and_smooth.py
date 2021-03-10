# NOTE: 256GB CPU memory required to run this script.

import os.path as osp
import time
import argparse

import torch
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_correction_layers', type=int, default=3), # tuned for 2, 3
    parser.add_argument('--correction_alpha', type=float, default=1.0), # tuned for 0.6, 0.8, 1.0
    parser.add_argument('--num_smoothing_layers', type=int, default=2), # tuned for 2, 3
    parser.add_argument('--smoothing_alpha', type=float, default=0.8), # tuned for 0.6, 0.8, 1.0
    args = parser.parse_args()
    print(args)

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    t = time.perf_counter()
    print('Reading graph-agnostic predictions...', end=' ', flush=True)
    y_pred = torch.from_numpy(np.load('results/cs/pred.npy'))
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Reading adjacency matrix...', end=' ', flush=True)
    path = dataset.root + '/mag240m/paper_to_paper_symmetric.pt'
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
    print('computing gcn_norm...')
    adj_t = gcn_norm(adj_t, add_self_loops=False)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    valid_idx = torch.from_numpy(dataset.get_idx_split('valid'))
    test_idx = torch.from_numpy(dataset.get_idx_split('test'))

    y_train = torch.from_numpy(dataset.paper_label[train_idx]).to(torch.long)
    y_valid = torch.from_numpy(dataset.paper_label[valid_idx]).to(torch.long)

    model = CorrectAndSmooth(args.num_correction_layers, args.correction_alpha,
                             args.num_smoothing_layers, args.smoothing_alpha, autoscale = True)

    t = time.perf_counter()
    print('Correcting predictions...', end=' ', flush=True)
    y_pred = model.correct(y_pred, y_train, train_idx, adj_t)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Smoothing predictions...', end=' ', flush=True)
    y_pred = model.smooth(y_pred, y_train, train_idx, adj_t)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    

    train_acc = evaluator.eval({
        'y_true': y_train,
        'y_pred': y_pred[train_idx].argmax(dim=-1)
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_valid,
        'y_pred': y_pred[valid_idx].argmax(dim=-1)
    })['acc']
    print(f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}')

    res = {'y_pred': y_pred[test_idx].argmax(dim=-1)}
    evaluator.save_test_submission(res, 'results/cs')
