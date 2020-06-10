import pandas as pd
import os.path as osp
import os
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from tqdm import tqdm

### reading raw files from a directory.
### for homogeneous graph
def read_csv_graph_raw(raw_dir, add_inverse_edge = True, additional_node_files = [], additional_edge_files = []):
    '''
    raw_dir: path to the raw directory
    add_inverse_edge (bool): whether to add inverse edge or not

    return: graph_list, which is a list of graphs.
    Each graph is a dictionary, containing edge_index, edge_feat, node_feat, and num_nodes
    edge_feat and node_feat are optional: if a graph does not contain it, we will have None.

    additional_node_files and additional_edge_files must be in the raw directory.
    - The name should be {additional_node_file, additional_edge_file}.csv.gz
    - The length should be num_nodes or num_edges

    
    '''

    print('Loading necessary files...')
    print('This might take a while.')
    # loading necessary files
    try:
        edge = pd.read_csv(osp.join(raw_dir, "edge.csv.gz"), compression="gzip", header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
        num_node_list = pd.read_csv(osp.join(raw_dir, "num-node-list.csv.gz"), compression="gzip", header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
        num_edge_list = pd.read_csv(osp.join(raw_dir, "num-edge-list.csv.gz"), compression="gzip", header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list

    except FileNotFoundError:
        raise RuntimeError("No necessary file")

    try:
        node_feat = pd.read_csv(osp.join(raw_dir, "node-feat.csv.gz"), compression="gzip", header = None).values
        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
    except FileNotFoundError:
        node_feat = None

    try:
        edge_feat = pd.read_csv(osp.join(raw_dir, "edge-feat.csv.gz"), compression="gzip", header = None).values
        if 'int' in str(edge_feat.dtype):
            edge_feat = edge_feat.astype(np.int64)
        else:
            #float
            edge_feat = edge_feat.astype(np.float32)

    except FileNotFoundError:
        edge_feat = None


    additional_node_info = {}   
    for additional_file in additional_node_files:
        temp = pd.read_csv(osp.join(raw_dir, additional_file + ".csv.gz"), compression="gzip", header = None).values

        if 'node_' not in additional_file:
            feat_name = 'node_' + additional_file
        else:
            feat_name = additional_file

        if 'int' in str(temp.dtype):
            additional_node_info[feat_name] = temp.astype(np.int64)
        else:
            # float
            additional_node_info[feat_name] = temp.astype(np.float32)

    additional_edge_info = {}   
    for additional_file in additional_edge_files:
        temp = pd.read_csv(osp.join(raw_dir, additional_file + ".csv.gz"), compression="gzip", header = None).values

        if 'edge_' not in additional_file:
            feat_name = 'edge_' + additional_file
        else:
            feat_name = additional_file

        if 'int' in str(temp.dtype):
            additional_edge_info[feat_name] = temp.astype(np.int64)
        else:
            # float
            additional_edge_info[feat_name] = temp.astype(np.float32)


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

            for key, value in additional_edge_info.items():
                graph[key] = np.repeat(value[num_edge_accum:num_edge_accum+num_edge], 2, axis = 0)

        else:
            graph["edge_index"] = edge[:, num_edge_accum:num_edge_accum+num_edge]

            if edge_feat is not None:
                graph["edge_feat"] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
            else:
                graph["edge_feat"] = None

            for key, value in additional_edge_info.items():
                graph[key] = value[num_edge_accum:num_edge_accum+num_edge]

        num_edge_accum += num_edge

        ### handling node
        if node_feat is not None:
            graph["node_feat"] = node_feat[num_node_accum:num_node_accum+num_node]
        else:
            graph["node_feat"] = None

        for key, value in additional_node_info.items():
            graph[key] = value[num_node_accum:num_node_accum+num_node]


        graph["num_nodes"] = num_node
        num_node_accum += num_node

        graph_list.append(graph)

    return graph_list


### reading raw files from a directory.
### for heterogeneous graph
def read_csv_heterograph_raw(raw_dir, add_inverse_edge = False, additional_node_files = [], additional_edge_files = []):
    '''
    raw_dir: path to the raw directory
    add_inverse_edge (bool): whether to add inverse edge or not

    return: graph_list, which is a list of heterogeneous graphs.
    Each graph is a dictionary, containing the following keys:
    - edge_index_dict
        edge_index_dict[(head, rel, tail)] = edge_index for (head, rel, tail)

    - edge_feat_dict
        edge_feat_dict[(head, rel, tail)] = edge_feat for (head, rel, tail)

    - node_feat_dict
        node_feat_dict[nodetype] = node_feat for nodetype
    
    - num_nodes_dict
        num_nodes_dict[nodetype] = num_nodes for nodetype
    
    * edge_feat_dict and node_feat_dict are optional: if a graph does not contain it, we will simply have None.

    We can also have additional node/edge features. For example,
    - edge_reltype_dict
        edge_reltype_dict[(head, rel, tail)] = edge_reltype for (head, rel, tail)

    - node_year_dict
        node_year_dict[nodetype] = node_year
    
    '''

    print('Loading necessary files...')
    print('This might take a while.')

    # loading necessary files
    try:
        num_node_df = pd.read_csv(osp.join(raw_dir, "num-node-dict.csv.gz"), compression="gzip")
        num_node_dict = {nodetype: num_node_df[nodetype].astype(np.int64).tolist() for nodetype in num_node_df.keys()}
        nodetype_list = sorted(list(num_node_dict.keys()))

        ## read edge_dict, num_edge_dict
        triplet_df = pd.read_csv(osp.join(raw_dir, "triplet-type-list.csv.gz"), compression="gzip", header = None)
        triplet_list = sorted([(head, relation, tail) for head, relation, tail in zip(triplet_df[0].tolist(), triplet_df[1].tolist(), triplet_df[2].tolist())])

        edge_dict = {}
        num_edge_dict = {}

        for triplet in triplet_list:
            subdir = osp.join(raw_dir, "relations", "___".join(triplet))

            edge_dict[triplet] = pd.read_csv(osp.join(subdir, "edge.csv.gz"), compression="gzip", header = None).values.T.astype(np.int64)
            num_edge_dict[triplet] = pd.read_csv(osp.join(subdir, "num-edge-list.csv.gz"), compression="gzip", header = None).astype(np.int64)[0].tolist()

        # check the number of graphs coincide
        assert(len(num_node_dict[nodetype_list[0]]) == len(num_edge_dict[triplet_list[0]]))

        num_graphs = len(num_node_dict[nodetype_list[0]])

    except FileNotFoundError:
        raise RuntimeError("No necessary file")

    node_feat_dict = {}
    for nodetype in nodetype_list:
        subdir = osp.join(raw_dir, "node-feat", nodetype)
        
        try:
            node_feat = pd.read_csv(osp.join(subdir, "node-feat.csv.gz"), compression="gzip", header = None).values
            if 'int' in str(node_feat.dtype):
                node_feat = node_feat.astype(np.int64)
            else:
                # float
                node_feat = node_feat.astype(np.float32)

            node_feat_dict[nodetype] = node_feat
        except FileNotFoundError:
            pass

    edge_feat_dict = {}
    for triplet in triplet_list:
        subdir = osp.join(raw_dir, "relations", "___".join(triplet))

        try:
            edge_feat = pd.read_csv(osp.join(subdir, "edge-feat.csv.gz"), compression="gzip", header = None).values
            if 'int' in str(edge_feat.dtype):
                edge_feat = edge_feat.astype(np.int64)
            else:
                #float
                edge_feat = edge_feat.astype(np.float32)

            edge_feat_dict[triplet] = edge_feat

        except FileNotFoundError:
            pass


    additional_node_info = {}
    # e.g., additional_node_info['node_year'] = node_feature_dict for node_year
    for additional_file in additional_node_files:
        additional_feat_dict = {}
        for nodetype in nodetype_list:
            subdir = osp.join(raw_dir, "node-feat", nodetype)
            
            try:
                node_feat = pd.read_csv(osp.join(subdir, additional_file + ".csv.gz"), compression="gzip", header = None).values
                if 'int' in str(node_feat.dtype):
                    node_feat = node_feat.astype(np.int64)
                else:
                    # float
                    node_feat = node_feat.astype(np.float32)

                assert(len(node_feat) == sum(num_node_dict[nodetype]))

                additional_feat_dict[nodetype] = node_feat

            except FileNotFoundError:
                pass

        if 'node_' not in additional_file:
            feat_name = 'node_' + additional_file
        else:
            feat_name = additional_file

        additional_node_info[feat_name] = additional_feat_dict

    additional_edge_info = {}
    # e.g., additional_edge_info['edge_reltype'] = edge_feat_dict for edge_reltype
    for additional_file in additional_edge_files:
        additional_feat_dict = {}
        for triplet in triplet_list:
            subdir = osp.join(raw_dir, "relations", "___".join(triplet))
            
            try:
                edge_feat = pd.read_csv(osp.join(subdir, additional_file + ".csv.gz"), compression="gzip", header = None).values
                if 'int' in str(edge_feat.dtype):
                    edge_feat = edge_feat.astype(np.int64)
                else:
                    # float
                    edge_feat = edge_feat.astype(np.float32)

                assert(len(edge_feat) == sum(num_edge_dict[triplet]))

                additional_feat_dict[triplet] = edge_feat

            except FileNotFoundError:
                pass

        if 'edge_' not in additional_file:
            feat_name = 'edge_' + additional_file
        else:
            feat_name = additional_file

        additional_edge_info[feat_name] = additional_feat_dict

    graph_list = []
    num_node_accum_dict = {nodetype: 0 for nodetype in nodetype_list}
    num_edge_accum_dict = {triplet: 0 for triplet in triplet_list}

    print('Processing graphs...')
    for i in tqdm(range(num_graphs)):

        graph = dict()

        ### set up default atribute
        graph["edge_index_dict"] = {}
        graph["edge_feat_dict"] = {}
        graph["node_feat_dict"] = {}
        graph["num_nodes_dict"] = {}

        ### set up additional node/edge attributes
        for key in additional_node_info.keys():
            graph[key] = {}

        for key in additional_edge_info.keys():
            graph[key] = {}

        ### handling edge
        for triplet in triplet_list:
            edge = edge_dict[triplet]
            num_edge = num_edge_dict[triplet][i]
            num_edge_accum = num_edge_accum_dict[triplet]

            if add_inverse_edge:
                ### add edge_index
                # duplicate edge
                duplicated_edge = np.repeat(edge[:, num_edge_accum:num_edge_accum + num_edge], 2, axis = 1)
                duplicated_edge[0, 1::2] = duplicated_edge[1,0::2]
                duplicated_edge[1, 1::2] = duplicated_edge[0,0::2]
                graph["edge_index_dict"][triplet] = duplicated_edge

                ### add default edge feature
                if len(edge_feat_dict) > 0:
                    # if edge_feat exists for some triplet
                    if triplet in edge_feat_dict:
                        graph["edge_feat_dict"][triplet] = np.repeat(edge_feat_dict[triplet][num_edge:num_edge + num_edge], 2, axis = 0)

                else:
                    # if edge_feat is not given for any triplet
                    graph["edge_feat_dict"] = None

                ### add additional edge feature
                for key, value in additional_edge_info.items():
                    if triplet in value:
                        graph[key][triplet] = np.repeat(value[triplet][num_edge_accum : num_edge_accum + num_edge], 2, axis = 0)

            else:
                ### add edge_index
                graph["edge_index_dict"][triplet] = edge[:, num_edge_accum:num_edge_accum+num_edge]

                ### add default edge feature
                if len(edge_feat_dict) > 0:
                    # if edge_feat exists for some triplet
                    if triplet in edge_feat_dict:
                        graph["edge_feat_dict"][triplet] = edge_feat_dict[triplet][num_edge:num_edge + num_edge]

                else:
                    # if edge_feat is not given for any triplet
                    graph["edge_feat_dict"] = None

                ### add additional edge feature
                for key, value in additional_edge_info.items():
                    if triplet in value:
                        graph[key][triplet] = value[triplet][num_edge_accum : num_edge_accum + num_edge]

            num_edge_accum_dict[triplet] += num_edge

        ### handling node
        for nodetype in nodetype_list:
            num_node = num_node_dict[nodetype][i]
            num_node_accum = num_node_accum_dict[nodetype]

            ### add default node feature
            if len(node_feat_dict) > 0:
                # if node_feat exists for some node type
                if nodetype in node_feat_dict:
                    graph["node_feat_dict"][nodetype] = node_feat_dict[nodetype][num_node_accum:num_node_accum + num_node]
            
            else:
                graph["node_feat_dict"] = None 

            ### add additional node feature
            for key, value in additional_node_info.items():
                if nodetype in value:
                    graph[key][nodetype] = value[nodetype][num_node_accum : num_node_accum + num_node]

            graph["num_nodes_dict"][nodetype] = num_node
            num_node_accum_dict[nodetype] += num_node

        graph_list.append(graph)

    return graph_list

def read_node_label_hetero(raw_dir):
    df = pd.read_csv(osp.join(raw_dir, 'nodetype-has-label.csv.gz'))
    label_dict = {}
    for nodetype in df.keys():
        has_label = df[nodetype].values[0]
        if has_label:
            label_dict[nodetype] = pd.read_csv(osp.join(raw_dir, "node-label", nodetype, "node-label.csv.gz"), compression="gzip", header = None).values

    if len(label_dict) == 0:
        raise RuntimeError('No node label file found.')

    return label_dict


def read_nodesplitidx_split_hetero(split_dir):
    df = pd.read_csv(osp.join(split_dir, 'nodetype-has-split.csv.gz'))
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    for nodetype in df.keys():
        has_label = df[nodetype].values[0]
        if has_label:
            train_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, "train.csv.gz"), compression="gzip", header = None).values.T[0]
            valid_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, "valid.csv.gz"), compression="gzip", header = None).values.T[0]
            test_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, "test.csv.gz"), compression="gzip", header = None).values.T[0]

    if len(train_dict) == 0:
        raise RuntimeError('No split file found.')

    return train_dict, valid_dict, test_dict

if __name__ == "__main__":
    pass


