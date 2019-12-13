import pandas as pd
import torch
from torch_geometric.data import Data
import os.path as osp
import numpy as np

def read_csv_graph_pyg(raw_dir, raw_file_names, add_inverse_edge):
    data = Data()

    for raw_file_name in raw_file_names:
        print(raw_file_name)
        mat = pd.read_csv(osp.join(raw_dir, raw_file_name), compression="gzip", header = None).values

        if raw_file_name == 'edge.csv.gz':
            if add_inverse_edge:
                existing_edge = mat.transpose()
                opposite_edge = np.zeros_like(existing_edge)
                opposite_edge[0] = existing_edge[1]
                opposite_edge[1] = existing_edge[0]
                bidirectional_edge = np.concatenate((existing_edge, opposite_edge), axis = 1)
                data.edge_index = torch.tensor(bidirectional_edge, dtype = torch.long).contiguous()
            else:
                data.edge_index = torch.tensor(mat.transpose(), dtype = torch.long).contiguous()

        elif raw_file_name == 'edge-feat.csv.gz':
            if add_inverse_edge:
                bidirectional_edge_attr = np.concatenate((mat, mat), axis = 0)
                data.edge_attr = torch.tensor(bidirectional_edge_attr, dtype = torch.float)
            else:
                data.edge_attr = torch.tensor(mat, dtype = torch.float)
        elif raw_file_name == 'node-feat.csv.gz':
            data.x = torch.tensor(mat, dtype = torch.float)
        else:
            raise ValueError("Unknown data file called " + raw_file_name)

    return data