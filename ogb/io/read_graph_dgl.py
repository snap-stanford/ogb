import pandas as pd
import torch
import os.path as osp
import numpy as np
import dgl
from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw
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
            g.edata["feat"] = torch.from_numpy(graph["edge_feat"])

        if graph["node_feat"] is not None:
            g.ndata["feat"] = torch.from_numpy(graph["node_feat"])

        for key in additional_node_files:
            if 'node_' not in key:
                feat_name = 'node_' + key
            else:
                feat_name = key
            g.ndata[feat_name] = torch.from_numpy(graph[feat_name])

        for key in additional_edge_files:
            if 'edge_' not in key:
                feat_name = 'edge_' + key
            else:
                feat_name = key
            g.edata[feat_name] = torch.from_numpy(graph[feat_name])

        dgl_graph_list.append(g)

    return dgl_graph_list


def read_csv_heterograph_dgl(raw_dir, add_inverse_edge = False, additional_node_files = [], additional_edge_files = []):

    graph_list = read_csv_heterograph_raw(raw_dir, add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)
    dgl_graph_list = []

    print('Converting graphs into DGL objects...')

    for graph in tqdm(graph_list):
        g_dict = {}

        # add edge connectivity
        for triplet, edge_index in graph["edge_index_dict"].items():
            edge_tuple = [(i, j) for i, j in zip(graph["edge_index_dict"][triplet][0], graph["edge_index_dict"][triplet][1])]
            g_dict[triplet] = edge_tuple

        dgl_hetero_graph = dgl.heterograph(g_dict)

        if graph["edge_feat_dict"] is not None:
            for triplet in graph["edge_feat_dict"].keys():
                dgl_hetero_graph.edges[triplet].data["feat"] = torch.from_numpy(graph["edge_feat_dict"][triplet])

        if graph["node_feat_dict"] is not None:
            for nodetype in graph["node_feat_dict"].keys():
                dgl_hetero_graph.nodes[nodetype].data["feat"] = torch.from_numpy(graph["node_feat_dict"][nodetype])

        for key in additional_node_files:
            if 'node_' not in key:
                feat_name = 'node_' + key
            else:
                feat_name = key

            for triplet in graph[feat_name].keys():
                dgl_hetero_graph.edges[triplet].data[feat_name] = torch.from_numpy(graph[feat_name][triplet])

            for nodetype in graph[feat_name].keys():
                dgl_hetero_graph.nodes[nodetype].data[feat_name] = torch.from_numpy(graph[feat_name][nodetype])

        for key in additional_edge_files:
            if 'edge_' not in key:
                feat_name = 'edge_' + key
            else:
                feat_name = key

            for triplet in graph[feat_name].keys():
                dgl_hetero_graph.edges[triplet].data[feat_name] = torch.from_numpy(graph[feat_name][triplet])

        dgl_graph_list.append(dgl_hetero_graph)


    return dgl_graph_list


if __name__ == "__main__":
    pass
