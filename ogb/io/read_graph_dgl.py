import pandas as pd
import torch
import os.path as osp
import numpy as np
import dgl
from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw, read_binary_graph_raw, read_binary_heterograph_raw
from tqdm import tqdm

def read_graph_dgl(raw_dir, add_inverse_edge = False, additional_node_files = [], additional_edge_files = [], binary=False):

    if binary:
        # npz
        graph_list = read_binary_graph_raw(raw_dir, add_inverse_edge)
    else:
        # csv
        graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)
        
    dgl_graph_list = []

    print('Converting graphs into DGL objects...')
    
    for graph in tqdm(graph_list):
        g = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes = graph['num_nodes'])

        if graph['edge_feat'] is not None:
            g.edata['feat'] = torch.from_numpy(graph['edge_feat'])

        if graph['node_feat'] is not None:
            g.ndata['feat'] = torch.from_numpy(graph['node_feat'])

        for key in additional_node_files:
            g.ndata[key[5:]] = torch.from_numpy(graph[key])

        for key in additional_edge_files:
            g.edata[key[5:]] = torch.from_numpy(graph[key])

        dgl_graph_list.append(g)

    return dgl_graph_list


def read_heterograph_dgl(raw_dir, add_inverse_edge = False, additional_node_files = [], additional_edge_files = [], binary=False):

    if binary:
        # npz
        graph_list = read_binary_heterograph_raw(raw_dir, add_inverse_edge)
    else:
        # csv
        graph_list = read_csv_heterograph_raw(raw_dir, add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)

    dgl_graph_list = []

    print('Converting graphs into DGL objects...')

    for graph in tqdm(graph_list):
        g_dict = {}

        # add edge connectivity
        for triplet, edge_index in graph['edge_index_dict'].items():
            edge_tuple = [(i, j) for i, j in zip(graph['edge_index_dict'][triplet][0], graph['edge_index_dict'][triplet][1])]
            g_dict[triplet] = edge_tuple

        dgl_hetero_graph = dgl.heterograph(g_dict, num_nodes_dict = graph['num_nodes_dict'])

        if graph['edge_feat_dict'] is not None:
            for triplet in graph['edge_feat_dict'].keys():
                dgl_hetero_graph.edges[triplet].data['feat'] = torch.from_numpy(graph['edge_feat_dict'][triplet])

        if graph['node_feat_dict'] is not None:
            for nodetype in graph['node_feat_dict'].keys():
                dgl_hetero_graph.nodes[nodetype].data['feat'] = torch.from_numpy(graph['node_feat_dict'][nodetype])

        for key in additional_node_files:
            for nodetype in graph[key].keys():
                dgl_hetero_graph.nodes[nodetype].data[key[5:]] = torch.from_numpy(graph[key][nodetype])

        for key in additional_edge_files:
            for triplet in graph[key].keys():
                dgl_hetero_graph.edges[triplet].data[key[5:]] = torch.from_numpy(graph[key][triplet])

        dgl_graph_list.append(dgl_hetero_graph)


    return dgl_graph_list


if __name__ == '__main__':
    pass
