import pandas as pd
import torch
from torch_geometric.data import Data
import os.path as osp
import numpy as np
from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw
from tqdm import tqdm

def read_csv_graph_pyg(raw_dir, add_inverse_edge = True, additional_node_files = [], additional_edge_files = []):

    graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)
    pyg_graph_list = []

    print('Converting graphs into PyG objects...')

    for graph in tqdm(graph_list):
        g = Data()
        g.__num_nodes__ = graph["num_nodes"]
        g.edge_index = torch.from_numpy(graph["edge_index"])

        if graph["edge_feat"] is not None:
            g.edge_attr = torch.from_numpy(graph["edge_feat"])

        if graph["node_feat"] is not None:
            g.x = torch.from_numpy(graph["node_feat"])

        for key in additional_node_files:
            if 'node_' not in key:
                feat_name = 'node_' + key
            else:
                feat_name = key
            g[feat_name] = torch.from_numpy(graph[feat_name])

        for key in additional_edge_files:
            if 'edge_' not in key:
                feat_name = 'edge_' + key
            else:
                feat_name = key
            g[feat_name] = torch.from_numpy(graph[feat_name])

        pyg_graph_list.append(g)

    return pyg_graph_list


def read_csv_heterograph_pyg(raw_dir, add_inverse_edge = False, additional_node_files = [], additional_edge_files = []):

    graph_list = read_csv_heterograph_raw(raw_dir, add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)
    pyg_graph_list = []

    print('Converting graphs into PyG objects...')

    for graph in tqdm(graph_list):
        g = Data()
        
        g.__num_nodes__ = graph["num_nodes_dict"]
        g.num_nodes_dict = graph["num_nodes_dict"]

        # add edge connectivity
        g.edge_index_dict = {}
        for triplet, edge_index in graph["edge_index_dict"].items():
            g.edge_index_dict[triplet] = torch.from_numpy(edge_index)

        if graph["edge_feat_dict"] is not None:
            g.edge_attr_dict = {}
            for triplet in graph["edge_feat_dict"].keys():
                g.edge_attr_dict[triplet] = torch.from_numpy(graph["edge_feat_dict"][triplet])

        if graph["node_feat_dict"] is not None:
            g.x_dict = {}
            for nodetype in graph["node_feat_dict"].keys():
                g.x_dict[nodetype] = torch.from_numpy(graph["node_feat_dict"][nodetype])


        for key in additional_node_files:
            if 'node_' not in key:
                feat_name = 'node_' + key
            else:
                feat_name = key

            g[feat_name + '_dict'] = {}
            for nodetype in graph[feat_name].keys():
                g[feat_name + '_dict'][nodetype] = torch.from_numpy(graph[feat_name][nodetype])

        for key in additional_edge_files:

            if 'edge_' not in key:
                feat_name = 'edge_' + key
            else:
                feat_name = key

            g[feat_name + '_dict'] = {}
            for triplet in graph[feat_name].keys():
                g[feat_name + '_dict'][triplet] = torch.from_numpy(graph[feat_name][triplet])

        pyg_graph_list.append(g)


    return pyg_graph_list

if __name__ == "__main__":
    pass