import pandas as pd
import torch
from torch_geometric.data import Data
import os.path as osp
import numpy as np
from ogb.io.read_graph_raw import read_csv_graph_raw
from tqdm import tqdm

def read_csv_graph_pyg(raw_dir, add_inverse_edge = False):

    graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge)
    pyg_graph_list = []

    print('Converting graphs into PyG objects...')
    for graph in tqdm(graph_list):
        g = Data()
        g.__num_nodes__ = graph["num_nodes"]
        g.edge_index = torch.tensor(graph["edge_index"])

        if graph["edge_feat"] is not None:
            g.edge_attr = torch.tensor(graph["edge_feat"])

        if graph["node_feat"] is not None:
            g.x = torch.tensor(graph["node_feat"])

        pyg_graph_list.append(g)

    return pyg_graph_list



if __name__ == "__main__":
    #graph_list = read_csv_graph_pyg('dataset/proteinfunc_v2/raw', add_inverse_edge = True)
    graph_list = read_csv_graph_pyg('dataset/tox21/raw', add_inverse_edge = True)

    print(graph_list)