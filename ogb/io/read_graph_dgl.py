import pandas as pd
import torch
import os.path as osp
import numpy as np
import dgl

def read_csv_graph_dgl(raw_dir, raw_file_names, add_inverse_edge):
    g = dgl.DGLGraph()

    print("edge.csv.gz")
    ### add nodes and edges
    edge_mat = pd.read_csv(osp.join(raw_dir, "edge.csv.gz"), compression="gzip", header = None).values # (num_edges, 2) numpy array
    num_nodes = np.max(edge_mat) + 1

    g.add_nodes(num_nodes)
    g.add_edges(edge_mat[:,0], edge_mat[:,1])

    if add_inverse_edge:
        g.add_edges(edge_mat[:,1], edge_mat[:,0])


    for raw_file_name in raw_file_names:
        ### skip edge.csv.gz
        if raw_file_name == "edge.csv.gz":
            continue

        print(raw_file_name)
        mat = pd.read_csv(osp.join(raw_dir, raw_file_name), compression="gzip", header = None).values

        if raw_file_name == 'edge-feat.csv.gz':
            if add_inverse_edge:
                bidirectional_edge_attr = np.concatenate((mat, mat), axis = 0)
                g.edata['feat'] = torch.tensor(bidirectional_edge_attr, dtype = torch.float)
            else:
                g.edata['feat'] = torch.tensor(mat, dtype = torch.float)
        elif raw_file_name == 'node-feat.csv.gz':
            g.ndata['feat'] = torch.tensor(mat, dtype = torch.float)
        else:
            raise ValueError("Unknown data file called " + raw_file_name)

    return g
