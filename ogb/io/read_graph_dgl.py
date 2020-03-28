import pandas as pd
import torch
import os.path as osp
import numpy as np
import dgl
from ogb.io.read_graph_raw import read_csv_graph_raw
from tqdm import tqdm

def read_csv_graph_dgl(raw_dir, add_inverse_edge = True, additional_node_files = [], additional_edge_files = []):

    graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)
    dgl_graph_list = []

    print('Converting graphs into DGL objects...')
    for graph in tqdm(graph_list):
        g = dgl.DGLGraph()
        g.add_nodes(graph["num_nodes"])
        g.add_edges(graph["edge_index"][0], graph["edge_index"][1])

        if graph["edge_feat"] is not None:
            g.edata["feat"] = torch.tensor(graph["edge_feat"])

        if graph["node_feat"] is not None:
            g.ndata["feat"] = torch.tensor(graph["node_feat"])

        for key in additional_node_files:
            g.ndata[key] = torch.tensor(graph[key])

        for key in additional_edge_files:
            g.edata[key] = torch.tensor(graph[key])

        dgl_graph_list.append(g)

    return dgl_graph_list

if __name__ == "__main__":
    # graph_list = read_csv_graph_dgl('dataset/proteinfunc_v2/raw', add_inverse_edge = True)
    graph_list = read_csv_graph_dgl('dataset/tox21/raw', add_inverse_edge = True)

    #print(graph_list)
