import pandas as pd
import os.path as osp
import os
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from tqdm import tqdm

### reading raw files from a directory.
def read_csv_graph_raw(raw_dir, add_inverse_edge = False):
    '''
    raw_dir: path to the raw directory
    add_inverse_edge (bool): whether to add inverse edge or not

    return: graph_list, which is a list of graphs.
    Each graph is a dictionary, containing edge_index, edge_feat, node_feat, and num_nodes
    edge_feat and node_feat are optional: if a graph does not contain it, we will have None.
    '''

    print('Loading necessary files...')
    # loading necessary files
    try:
        edge = pd.read_csv(osp.join(raw_dir, "edge.csv.gz"), compression="gzip", header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
        num_node_list = pd.read_csv(osp.join(raw_dir, "num-node-list.csv.gz"), compression="gzip", header = None).values.T[0].astype(np.int64).tolist() # (num_graph, ) python list
        num_edge_list = pd.read_csv(osp.join(raw_dir, "num-edge-list.csv.gz"), compression="gzip", header = None).values.T[0].astype(np.int64).tolist() # (num_edge, ) python list
    except:
        raise RuntimeError("No necessary file")

    try:
        node_feat = pd.read_csv(osp.join(raw_dir, "node-feat.csv.gz"), compression="gzip", header = None).values
        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
    except:
        node_feat = None

    try:
        edge_feat = pd.read_csv(osp.join(raw_dir, "edge-feat.csv.gz"), compression="gzip", header = None).values
        if 'int' in str(edge_feat.dtype):
            edge_feat = edge_feat.astype(np.int64)
        else:
            #float
            edge_feat = edge_feat.astype(np.float32)

    except:
        edge_feat = None

    graph_list = []
    num_node_accum = 0
    num_edge_accum = 0

    print('Processing graphs...')
    for num_node, num_edge in tqdm(zip(num_node_list, num_edge_list), total=len(num_node_list)):

        graph = dict()

        ### handling edge
        if add_inverse_edge:
            ### duplicate edge
            duplicated_edge = np.repeat(edge[:, num_edge_accum:num_edge_accum+num_edge], 2, axis = 1)
            duplicated_edge[0, 1::2] = duplicated_edge[1,0::2]
            duplicated_edge[1, 1::2] = duplicated_edge[0,0::2]

            graph["edge_index"] = duplicated_edge

            if edge_feat is not None:
                graph["edge_feat"] = np.repeat(edge_feat[num_edge_accum:num_edge_accum+num_edge], 2, axis = 0)
            else:
                graph["edge_feat"] = None

        else:
            graph["edge_index"] = edge[:, num_edge_accum:num_edge_accum+num_edge]

            if edge_feat is not None:
                graph["edge_feat"] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
            else:
                graph["edge_feat"] = None

        num_edge_accum += num_edge

        ### handling node
        if node_feat is not None:
            graph["node_feat"] = node_feat[num_node_accum:num_node_accum+num_node]
        else:
            graph["node_feat"] = None

        graph["num_nodes"] = num_node
        num_node_accum += num_node

        graph_list.append(graph)

    return graph_list



if __name__ == "__main__":
    pass


