import pandas as pd
import os.path as osp
import os
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from tqdm import tqdm

### reading raw files from a directory.
### for homogeneous graph
def read_csv_graph_raw(raw_dir, add_inverse_edge = False, additional_node_files = [], additional_edge_files = []):
    '''
    raw_dir: path to the raw directory
    add_inverse_edge (bool): whether to add inverse edge or not

    return: graph_list, which is a list of graphs.
    Each graph is a dictionary, containing edge_index, edge_feat, node_feat, and num_nodes
    edge_feat and node_feat are optional: if a graph does not contain it, we will have None.

    additional_node_files and additional_edge_files must be in the raw directory.
    - The name should be {additional_node_file, additional_edge_file}.csv.gz
    - The length should be num_nodes or num_edges

    additional_node_files must start from 'node_'
    additional_edge_files must start from 'edge_'

    
    '''

    print('Loading necessary files...')
    print('This might take a while.')
    # loading necessary files
    try:
        edge = pd.read_csv(osp.join(raw_dir, 'edge.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
        num_node_list = pd.read_csv(osp.join(raw_dir, 'num-node-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
        num_edge_list = pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list

    except FileNotFoundError:
        raise RuntimeError('No necessary file')

    try:
        node_feat = pd.read_csv(osp.join(raw_dir, 'node-feat.csv.gz'), compression='gzip', header = None).values
        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
    except FileNotFoundError:
        node_feat = None

    try:
        edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv.gz'), compression='gzip', header = None).values
        if 'int' in str(edge_feat.dtype):
            edge_feat = edge_feat.astype(np.int64)
        else:
            #float
            edge_feat = edge_feat.astype(np.float32)

    except FileNotFoundError:
        edge_feat = None


    additional_node_info = {}   
    for additional_file in additional_node_files:
        assert(additional_file[:5] == 'node_')

        # hack for ogbn-proteins
        if additional_file == 'node_species' and osp.exists(osp.join(raw_dir, 'species.csv.gz')):
            os.rename(osp.join(raw_dir, 'species.csv.gz'), osp.join(raw_dir, 'node_species.csv.gz'))

        temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), compression='gzip', header = None).values

        if 'int' in str(temp.dtype):
            additional_node_info[additional_file] = temp.astype(np.int64)
        else:
            # float
            additional_node_info[additional_file] = temp.astype(np.float32)

    additional_edge_info = {}   
    for additional_file in additional_edge_files:
        assert(additional_file[:5] == 'edge_')
        temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), compression='gzip', header = None).values

        if 'int' in str(temp.dtype):
            additional_edge_info[additional_file] = temp.astype(np.int64)
        else:
            # float
            additional_edge_info[additional_file] = temp.astype(np.float32)


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

            graph['edge_index'] = duplicated_edge

            if edge_feat is not None:
                graph['edge_feat'] = np.repeat(edge_feat[num_edge_accum:num_edge_accum+num_edge], 2, axis = 0)
            else:
                graph['edge_feat'] = None

            for key, value in additional_edge_info.items():
                graph[key] = np.repeat(value[num_edge_accum:num_edge_accum+num_edge], 2, axis = 0)

        else:
            graph['edge_index'] = edge[:, num_edge_accum:num_edge_accum+num_edge]

            if edge_feat is not None:
                graph['edge_feat'] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
            else:
                graph['edge_feat'] = None

            for key, value in additional_edge_info.items():
                graph[key] = value[num_edge_accum:num_edge_accum+num_edge]

        num_edge_accum += num_edge

        ### handling node
        if node_feat is not None:
            graph['node_feat'] = node_feat[num_node_accum:num_node_accum+num_node]
        else:
            graph['node_feat'] = None

        for key, value in additional_node_info.items():
            graph[key] = value[num_node_accum:num_node_accum+num_node]


        graph['num_nodes'] = num_node
        num_node_accum += num_node

        graph_list.append(graph)

    return graph_list


### reading raw files from a directory.
### npz ver
### for homogeneous graph
def read_binary_graph_raw(raw_dir, add_inverse_edge = False):
    '''
    raw_dir: path to the raw directory
    add_inverse_edge (bool): whether to add inverse edge or not

    return: graph_list, which is a list of graphs.
    Each graph is a dictionary, containing edge_index, edge_feat, node_feat, and num_nodes
    edge_feat and node_feat are optional: if a graph does not contain it, we will have None.

    raw_dir must contain data.npz
    - edge_index
    - num_nodes_list
    - num_edges_list
    - node_** (optional, node_feat is the default node features)
    - edge_** (optional, edge_feat is the default edge features)
    '''

    if add_inverse_edge:
        raise RuntimeError('add_inverse_edge is depreciated in read_binary')

    print('Loading necessary files...')
    print('This might take a while.')
    data_dict = np.load(osp.join(raw_dir, 'data.npz'))

    edge_index = data_dict['edge_index']
    num_nodes_list = data_dict['num_nodes_list']
    num_edges_list = data_dict['num_edges_list']

    # storing node and edge features
    node_dict = {}
    edge_dict = {}

    for key in list(data_dict.keys()):
        if key == 'edge_index' or key == 'num_nodes_list' or key == 'num_edges_list':
            continue

        if key[:5] == 'node_':
            node_dict[key] = data_dict[key]
        elif key[:5] == 'edge_':
            edge_dict[key] = data_dict[key]
        else:
            raise RuntimeError(f"Keys in graph object should start from either \'node_\' or \'edge_\', but found \'{key}\'.")

    graph_list = []
    num_nodes_accum = 0
    num_edges_accum = 0

    print('Processing graphs...')
    for num_nodes, num_edges in tqdm(zip(num_nodes_list, num_edges_list), total=len(num_nodes_list)):

        graph = dict()

        graph['edge_index'] = edge_index[:, num_edges_accum:num_edges_accum+num_edges]

        for key, feat in edge_dict.items():
            graph[key] = feat[num_edges_accum:num_edges_accum+num_edges]

        if 'edge_feat' not in graph:
            graph['edge_feat'] =  None

        for key, feat in node_dict.items():
            graph[key] = feat[num_nodes_accum:num_nodes_accum+num_nodes]

        if 'node_feat' not in graph:
            graph['node_feat'] = None

        graph['num_nodes'] = num_nodes

        num_edges_accum += num_edges
        num_nodes_accum += num_nodes

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
        num_node_df = pd.read_csv(osp.join(raw_dir, 'num-node-dict.csv.gz'), compression='gzip')
        num_node_dict = {nodetype: num_node_df[nodetype].astype(np.int64).tolist() for nodetype in num_node_df.keys()}
        nodetype_list = sorted(list(num_node_dict.keys()))

        ## read edge_dict, num_edge_dict
        triplet_df = pd.read_csv(osp.join(raw_dir, 'triplet-type-list.csv.gz'), compression='gzip', header = None)
        triplet_list = sorted([(head, relation, tail) for head, relation, tail in zip(triplet_df[0].tolist(), triplet_df[1].tolist(), triplet_df[2].tolist())])

        edge_dict = {}
        num_edge_dict = {}

        for triplet in triplet_list:
            subdir = osp.join(raw_dir, 'relations', '___'.join(triplet))

            edge_dict[triplet] = pd.read_csv(osp.join(subdir, 'edge.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64)
            num_edge_dict[triplet] = pd.read_csv(osp.join(subdir, 'num-edge-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist()

        # check the number of graphs coincide
        assert(len(num_node_dict[nodetype_list[0]]) == len(num_edge_dict[triplet_list[0]]))

        num_graphs = len(num_node_dict[nodetype_list[0]])

    except FileNotFoundError:
        raise RuntimeError('No necessary file')

    node_feat_dict = {}
    for nodetype in nodetype_list:
        subdir = osp.join(raw_dir, 'node-feat', nodetype)
        
        try:
            node_feat = pd.read_csv(osp.join(subdir, 'node-feat.csv.gz'), compression='gzip', header = None).values
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
        subdir = osp.join(raw_dir, 'relations', '___'.join(triplet))

        try:
            edge_feat = pd.read_csv(osp.join(subdir, 'edge-feat.csv.gz'), compression='gzip', header = None).values
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
        assert(additional_file[:5] == 'node_')

        for nodetype in nodetype_list:
            subdir = osp.join(raw_dir, 'node-feat', nodetype)

            try:
                node_feat = pd.read_csv(osp.join(subdir, additional_file + '.csv.gz'), compression='gzip', header = None).values
                if 'int' in str(node_feat.dtype):
                    node_feat = node_feat.astype(np.int64)
                else:
                    # float
                    node_feat = node_feat.astype(np.float32)

                assert(len(node_feat) == sum(num_node_dict[nodetype]))

                additional_feat_dict[nodetype] = node_feat

            except FileNotFoundError:
                pass

        additional_node_info[additional_file] = additional_feat_dict

    additional_edge_info = {}
    # e.g., additional_edge_info['edge_reltype'] = edge_feat_dict for edge_reltype
    for additional_file in additional_edge_files:
        assert(additional_file[:5] == 'edge_')
        additional_feat_dict = {}
        for triplet in triplet_list:
            subdir = osp.join(raw_dir, 'relations', '___'.join(triplet))
            
            try:
                edge_feat = pd.read_csv(osp.join(subdir, additional_file + '.csv.gz'), compression='gzip', header = None).values
                if 'int' in str(edge_feat.dtype):
                    edge_feat = edge_feat.astype(np.int64)
                else:
                    # float
                    edge_feat = edge_feat.astype(np.float32)

                assert(len(edge_feat) == sum(num_edge_dict[triplet]))

                additional_feat_dict[triplet] = edge_feat

            except FileNotFoundError:
                pass

        additional_edge_info[additional_file] = additional_feat_dict

    graph_list = []
    num_node_accum_dict = {nodetype: 0 for nodetype in nodetype_list}
    num_edge_accum_dict = {triplet: 0 for triplet in triplet_list}

    print('Processing graphs...')
    for i in tqdm(range(num_graphs)):

        graph = dict()

        ### set up default atribute
        graph['edge_index_dict'] = {}
        graph['edge_feat_dict'] = {}
        graph['node_feat_dict'] = {}
        graph['num_nodes_dict'] = {}

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
                graph['edge_index_dict'][triplet] = duplicated_edge

                ### add default edge feature
                if len(edge_feat_dict) > 0:
                    # if edge_feat exists for some triplet
                    if triplet in edge_feat_dict:
                        graph['edge_feat_dict'][triplet] = np.repeat(edge_feat_dict[triplet][num_edge:num_edge + num_edge], 2, axis = 0)

                else:
                    # if edge_feat is not given for any triplet
                    graph['edge_feat_dict'] = None

                ### add additional edge feature
                for key, value in additional_edge_info.items():
                    if triplet in value:
                        graph[key][triplet] = np.repeat(value[triplet][num_edge_accum : num_edge_accum + num_edge], 2, axis = 0)

            else:
                ### add edge_index
                graph['edge_index_dict'][triplet] = edge[:, num_edge_accum:num_edge_accum+num_edge]

                ### add default edge feature
                if len(edge_feat_dict) > 0:
                    # if edge_feat exists for some triplet
                    if triplet in edge_feat_dict:
                        graph['edge_feat_dict'][triplet] = edge_feat_dict[triplet][num_edge:num_edge + num_edge]

                else:
                    # if edge_feat is not given for any triplet
                    graph['edge_feat_dict'] = None

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
                    graph['node_feat_dict'][nodetype] = node_feat_dict[nodetype][num_node_accum:num_node_accum + num_node]
            
            else:
                graph['node_feat_dict'] = None 

            ### add additional node feature
            for key, value in additional_node_info.items():
                if nodetype in value:
                    graph[key][nodetype] = value[nodetype][num_node_accum : num_node_accum + num_node]

            graph['num_nodes_dict'][nodetype] = num_node
            num_node_accum_dict[nodetype] += num_node

        graph_list.append(graph)

    return graph_list


def read_binary_heterograph_raw(raw_dir, add_inverse_edge = False):
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
    - edge_**
    - node_**
    
    '''

    if add_inverse_edge:
        raise RuntimeError('add_inverse_edge is depreciated in read_binary')

    print('Loading necessary files...')
    print('This might take a while.')

    # loading necessary files
    try:
        num_nodes_dict = read_npz_dict(osp.join(raw_dir, 'num_nodes_dict.npz'))
        tmp = read_npz_dict(osp.join(raw_dir, 'num_edges_dict.npz'))
        num_edges_dict = {tuple(key.split('___')): tmp[key] for key in tmp.keys()}
        del tmp
        tmp = read_npz_dict(osp.join(raw_dir, 'edge_index_dict.npz'))
        edge_index_dict = {tuple(key.split('___')): tmp[key] for key in tmp.keys()}
        del tmp
        
        ent_type_list = sorted(list(num_nodes_dict.keys()))
        triplet_type_list = sorted(list(num_edges_dict.keys()))

        num_graphs = len(num_nodes_dict[ent_type_list[0]])

    except FileNotFoundError:
        raise RuntimeError('No necessary file')

    # storing node and edge features
    # mapping from the name of the features to feat_dict
    node_feat_dict_dict = {}
    edge_feat_dict_dict = {}

    for filename in os.listdir(raw_dir):
        if '.npz' not in filename:
            continue
        if filename in ['num_nodes_dict.npz', 'num_edges_dict.npz', 'edge_index_dict.npz']:
            continue

        # do not read target label information here
        if '-label.npz' in filename:
            continue

        feat_name = filename.split('.')[0]

        if 'node_' in feat_name:
            feat_dict = read_npz_dict(osp.join(raw_dir, filename))
            node_feat_dict_dict[feat_name] = feat_dict
        elif 'edge_' in feat_name:
            tmp = read_npz_dict(osp.join(raw_dir, filename))
            feat_dict = {tuple(key.split('___')): tmp[key] for key in tmp.keys()}
            del tmp
            edge_feat_dict_dict[feat_name] = feat_dict
        else:
            raise RuntimeError(f"Keys in graph object should start from either \'node_\' or \'edge_\', but found \'{feat_name}\'.")

    graph_list = []
    num_nodes_accum_dict = {ent_type: 0 for ent_type in ent_type_list}
    num_edges_accum_dict = {triplet: 0 for triplet in triplet_type_list}

    print('Processing graphs...')
    for i in tqdm(range(num_graphs)):

        graph = dict()

        ### set up default atribute
        graph['edge_index_dict'] = {}
        graph['num_nodes_dict'] = {}

        for feat_name in node_feat_dict_dict.keys():
            graph[feat_name] = {}

        for feat_name in edge_feat_dict_dict.keys():
            graph[feat_name] = {}

        if not 'edge_feat_dict' in graph:
            graph['edge_feat_dict'] = None

        if not 'node_feat_dict' in graph:
            graph['node_feat_dict'] = None

        ### handling edge
        for triplet in triplet_type_list:
            edge_index = edge_index_dict[triplet]
            num_edges = num_edges_dict[triplet][i]
            num_edges_accum = num_edges_accum_dict[triplet]

            ### add edge_index
            graph['edge_index_dict'][triplet] = edge_index[:, num_edges_accum:num_edges_accum+num_edges]

            ### add edge feature
            for feat_name in edge_feat_dict_dict.keys():
                if triplet in  edge_feat_dict_dict[feat_name]:
                    feat = edge_feat_dict_dict[feat_name][triplet]
                    graph[feat_name][triplet] = feat[num_edges_accum : num_edges_accum + num_edges]

            num_edges_accum_dict[triplet] += num_edges

        ### handling node
        for ent_type in ent_type_list:
            num_nodes = num_nodes_dict[ent_type][i]
            num_nodes_accum = num_nodes_accum_dict[ent_type]

            ### add node feature
            for feat_name in node_feat_dict_dict.keys():
                if ent_type in node_feat_dict_dict[feat_name]:
                    feat = node_feat_dict_dict[feat_name][ent_type]
                    graph[feat_name][ent_type] = feat[num_nodes_accum : num_nodes_accum + num_nodes]

            graph['num_nodes_dict'][ent_type] = num_nodes
            num_nodes_accum_dict[ent_type] += num_nodes

        graph_list.append(graph)

    return graph_list

def read_npz_dict(path):
    tmp = np.load(path)
    dict = {}
    for key in tmp.keys():
        dict[key] = tmp[key]
    del tmp
    return dict

def read_node_label_hetero(raw_dir):
    df = pd.read_csv(osp.join(raw_dir, 'nodetype-has-label.csv.gz'))
    label_dict = {}
    for nodetype in df.keys():
        has_label = df[nodetype].values[0]
        if has_label:
            label_dict[nodetype] = pd.read_csv(osp.join(raw_dir, 'node-label', nodetype, 'node-label.csv.gz'), compression='gzip', header = None).values

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
            train_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
            valid_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
            test_dict[nodetype] = pd.read_csv(osp.join(split_dir, nodetype, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

    if len(train_dict) == 0:
        raise RuntimeError('No split file found.')

    return train_dict, valid_dict, test_dict

if __name__ == '__main__':
    pass


