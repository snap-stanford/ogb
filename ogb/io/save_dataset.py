import torch
import pandas as pd
import os
import os.path as osp
from datetime import date
import shutil
from tqdm import tqdm
import numpy as np
from ogb.io.read_graph_raw import read_binary_graph_raw, read_binary_heterograph_raw
from ogb.utils.torch_util import all_numpy

class DatasetSaver(object):
    '''
        A class for saving graphs and split in OGB-compatible manner
        Create submission_datasetname/ directory, and output the following two files:
            - datasetname.zip (OGB-compatible zipped dataset folder)
            - meta_dict.pt (torch files storing all the necessary dataset meta-information)
    '''
    def __init__(self, dataset_name, is_hetero, version, root = 'submission'):
        # verify input
        if not ('ogbn-' in dataset_name or 'ogbl-' in dataset_name or 'ogbg-' in dataset_name):
            raise ValueError('Dataset name must have valid ogb prefix (e.g., ogbn-*).')
        if not isinstance(is_hetero, bool):
            raise ValueError('is_hetero must be of type bool.')
        if not (isinstance(version, int) and version >= 0):
            raise ValueError('version must be of type int and non-negative')

        self.dataset_name = dataset_name

        self.is_hetero = is_hetero
        self.version = version
        self.root = root
        self.dataset_prefix = dataset_name.split('-')[0] # specify the task category
        self.dataset_suffix = '_'.join(dataset_name.split('-')[1:])
        self.submission_dir = self.root + '_' + self.dataset_prefix + '_' + self.dataset_suffix
        self.dataset_dir = osp.join(self.submission_dir, self.dataset_suffix) 
        self.meta_dict_path = osp.join(self.submission_dir, 'meta_dict.pt')
        
        if self.dataset_prefix == 'ogbg' and self.is_hetero:
            raise NotImplementedError('Heterogeneous graph dataset object has not been implemented for graph property prediction yet.')

        if osp.exists(self.dataset_dir):
            if input(f'Found an existing submission directory at {self.submission_dir}/. \nWill you remove it? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.submission_dir)
                print('Removed existing submission directory')
            else:
                print('Process stopped.')
                exit(-1)


        # make necessary dirs
        self.raw_dir = osp.join(self.dataset_dir, 'raw')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(osp.join(self.dataset_dir, 'processed'), exist_ok=True)

        # create release note
        with open(osp.join(self.dataset_dir, f'RELEASE_v{version}.txt'), 'w') as fw:
            fw.write(f'# Release note for {self.dataset_name}\n\n### v{version}: {date.today()}')

        # check list
        self._save_graph_list_done = False
        self._save_split_done = False
        self._copy_mapping_dir_done = False

        if 'ogbl' == self.dataset_prefix:
            self._save_target_labels_done = True # for ogbl, we do not need to give predicted labels
        else:
            self._save_target_labels_done = False # for ogbn and ogbg, need to give predicted labels
        
        self._save_task_info_done = False
        self._get_meta_dict_done = False
        self._zip_done = False

    def _save_graph_list_hetero(self, graph_list):
        dict_keys = graph_list[0].keys()
        # check necessary keys
        if not 'edge_index_dict' in dict_keys:
            raise RuntimeError('edge_index_dict needs to be provided in graph objects')
        if not 'num_nodes_dict' in dict_keys:
            raise RuntimeError('num_nodes_dict needs to be provided in graph objects')

        print(dict_keys)

        # Store the following files
        # - edge_index_dict.npz (necessary)
        #   edge_index_dict
        # - num_nodes_dict.npz (necessary)
        #   num_nodes_dict
        # - num_edges_dict.npz (necessary)
        #   num_edges_dict
        # - node_**.npz (optional, node_feat_dict is the default node features)
        # - edge_**.npz (optional, edge_feat_dict the default edge features)
        
        # extract entity types
        ent_type_list = sorted([e for e in graph_list[0]['num_nodes_dict'].keys()])

        # saving num_nodes_dict
        print('Saving num_nodes_dict')
        num_nodes_dict = {}
        for ent_type in ent_type_list:
            num_nodes_dict[ent_type] = np.array([graph['num_nodes_dict'][ent_type] for graph in graph_list]).astype(np.int64)
        np.savez_compressed(osp.join(self.raw_dir, 'num_nodes_dict.npz'), **num_nodes_dict)
        
        print(num_nodes_dict)

        # extract triplet types
        triplet_type_list = sorted([(h, r, t) for (h, r, t) in graph_list[0]['edge_index_dict'].keys()])
        print(triplet_type_list)

        # saving edge_index_dict
        print('Saving edge_index_dict')
        num_edges_dict = {}
        edge_index_dict = {}
        for triplet in triplet_type_list:
            # representing triplet (head, rel, tail) as a single string 'head___rel___tail'
            triplet_cat = '___'.join(triplet)
            edge_index = np.concatenate([graph['edge_index_dict'][triplet] for graph in graph_list], axis = 1).astype(np.int64)
            if edge_index.shape[0] != 2:
                raise RuntimeError('edge_index must have shape (2, num_edges)')

            num_edges = np.array([graph['edge_index_dict'][triplet].shape[1] for graph in graph_list]).astype(np.int64)
            num_edges_dict[triplet_cat] = num_edges
            edge_index_dict[triplet_cat] = edge_index

        print(edge_index_dict)
        print(num_edges_dict)

        np.savez_compressed(osp.join(self.raw_dir, 'edge_index_dict.npz'), **edge_index_dict)
        np.savez_compressed(osp.join(self.raw_dir, 'num_edges_dict.npz'), **num_edges_dict)

        for key in dict_keys:
            if key == 'edge_index_dict' or key == 'num_nodes_dict':
                continue 
            if graph_list[0][key] is None:
                continue

            print(f'Saving {key}')

            feat_dict = {}

            if 'node_' in key:
                # node feature dictionary
                for ent_type in graph_list[0][key].keys():
                    if ent_type not in num_nodes_dict:
                        raise RuntimeError(f'Encountered unknown entity type called {ent_type}.')
                    
                    # check num_nodes
                    for i in range(len(graph_list)):
                        if len(graph_list[i][key][ent_type]) != num_nodes_dict[ent_type][i]:
                            raise RuntimeError(f'num_nodes mistmatches with {key}[{ent_type}]')
                    
                    # make sure saved in np.int64 or np.float32
                    dtype = np.int64 if 'int' in str(graph_list[0][key][ent_type].dtype) else np.float32
                    cat_feat = np.concatenate([graph[key][ent_type] for graph in graph_list], axis = 0).astype(dtype)
                    feat_dict[ent_type] = cat_feat
                
            elif 'edge_' in key:
                # edge feature dictionary
                for triplet in graph_list[0][key].keys():
                    # representing triplet (head, rel, tail) as a single string 'head___rel___tail'
                    triplet_cat = '___'.join(triplet)
                    if triplet_cat not in num_edges_dict:
                        raise RuntimeError(f"Encountered unknown triplet type called ({','.join(triplet)}).")

                    # check num_edges
                    for i in range(len(graph_list)):
                        if len(graph_list[i][key][triplet]) != num_edges_dict[triplet_cat][i]:
                            raise RuntimeError(f"num_edges mismatches with {key}[({','.join(triplet)})]")

                    # make sure saved in np.int64 or np.float32
                    dtype = np.int64 if 'int' in str(graph_list[0][key][triplet].dtype) else np.float32
                    cat_feat = np.concatenate([graph[key][triplet] for graph in graph_list], axis = 0).astype(dtype)
                    feat_dict[triplet_cat] = cat_feat

            else:
                raise RuntimeError(f'Keys in graph object should start from either \'node_\' or \'edge_\', but \'{key}\' given.')

            np.savez_compressed(osp.join(self.raw_dir, f'{key}.npz'), **feat_dict)

        print('Validating...')
        # testing
        print('Reading saved files')
        graph_list_read = read_binary_heterograph_raw(self.raw_dir, False)

        print('Checking read graphs and given graphs are the same')
        for i in tqdm(range(len(graph_list))):
            for key0, value0 in graph_list[i].items():
                if value0 is not None:
                    for key1, value1 in value0.items():
                        if isinstance(graph_list[i][key0][key1], np.ndarray):
                            assert(np.allclose(graph_list[i][key0][key1], graph_list_read[i][key0][key1], rtol=1e-04, atol=1e-04, equal_nan=True))
                        else:
                            assert(graph_list[i][key0][key1] == graph_list_read[i][key0][key1])

        del graph_list_read


    def _save_graph_list_homo(self, graph_list):        
        dict_keys = graph_list[0].keys()
        # check necessary keys
        if not 'edge_index' in dict_keys:
            raise RuntimeError('edge_index needs to be provided in graph objects')
        if not 'num_nodes' in dict_keys:
            raise RuntimeError('num_nodes needs to be provided in graph objects')

        print(dict_keys)

        data_dict = {}
        # Store the following keys
        # - edge_index (necessary)
        # - num_nodes_list (necessary)
        # - num_edges_list (necessary)
        # - node_** (optional, node_feat is the default node features)
        # - edge_** (optional, edge_feat is the default edge features)

        # saving num_nodes_list
        num_nodes_list = np.array([graph['num_nodes'] for graph in graph_list]).astype(np.int64)
        data_dict['num_nodes_list'] = num_nodes_list

        # saving edge_index and num_edges_list
        print('Saving edge_index')
        edge_index = np.concatenate([graph['edge_index'] for graph in graph_list], axis = 1).astype(np.int64)
        num_edges_list = np.array([graph['edge_index'].shape[1] for graph in graph_list]).astype(np.int64)

        if edge_index.shape[0] != 2:
            raise RuntimeError('edge_index must have shape (2, num_edges)')

        data_dict['edge_index'] = edge_index
        data_dict['num_edges_list'] = num_edges_list

        for key in dict_keys:
            if key == 'edge_index' or key == 'num_nodes':
                continue 
            if graph_list[0][key] is None:
                continue

            if 'node_' == key[:5]:
                # make sure saved in np.int64 or np.float32
                dtype = np.int64 if 'int' in str(graph_list[0][key].dtype) else np.float32
                # check num_nodes
                for i in range(len(graph_list)):
                    if len(graph_list[i][key]) != num_nodes_list[i]:
                        raise RuntimeError(f'num_nodes mistmatches with {key}')

                cat_feat = np.concatenate([graph[key] for graph in graph_list], axis = 0).astype(dtype)
                data_dict[key] = cat_feat

            elif 'edge_' == key[:5]:
                # make sure saved in np.int64 or np.float32
                dtype = np.int64 if 'int' in str(graph_list[0][key].dtype) else np.float32
                # check num_edges
                for i in range(len(graph_list)):
                    if len(graph_list[i][key]) != num_edges_list[i]:
                        raise RuntimeError(f'num_edges mistmatches with {key}')

                cat_feat = np.concatenate([graph[key] for graph in graph_list], axis = 0).astype(dtype)
                data_dict[key] = cat_feat

            else:
                raise RuntimeError(f'Keys in graph object should start from either \'node_\' or \'edge_\', but \'{key}\' given.')

        print('Saving all the files!')
        np.savez_compressed(osp.join(self.raw_dir, 'data.npz'), **data_dict)
        print('Validating...')
        # testing
        print('Reading saved files')
        graph_list_read = read_binary_graph_raw(self.raw_dir, False)

        print('Checking read graphs and given graphs are the same')
        for i in tqdm(range(len(graph_list))):
            # assert(graph_list[i].keys() == graph_list_read[i].keys())
            for key in graph_list[i].keys():
                if graph_list[i][key] is not None:
                    if isinstance(graph_list[i][key], np.ndarray):
                        assert(np.allclose(graph_list[i][key], graph_list_read[i][key], rtol=1e-4, atol=1e-4, equal_nan=True))
                    else:
                        assert(graph_list[i][key] == graph_list_read[i][key])

        del graph_list_read

    def save_task_info(self, task_type, eval_metric, num_classes = None):
        '''
            task_type (str): For ogbg and ogbn, either classification or regression.
            eval_metric (str): the metric
            if task_type is 'classification', num_classes must be given.
        '''

        if self.dataset_prefix == 'ogbn' or self.dataset_prefix == 'ogbg':
            if not ('classification' in task_type or 'regression' in task_type):
                raise ValueError(f'task type must contain eighther classification or regression, but {task_type} given')

        self.task_type = task_type

        print(self.task_type)
        print(num_classes)
        
        if 'classification' in self.task_type:
            if not (isinstance(num_classes, int) and num_classes > 1):
                raise ValueError(f'num_classes must be an integer larger than 1, {num_classes} given.')
            self.num_classes = num_classes
        else:
            self.num_classes = -1 # in the case of regression, just set to -1

        self.eval_metric = eval_metric

        self._save_task_info_done = True

    def save_target_labels(self, target_labels):
        '''
            target_label (numpy.narray): storing target labels. Shape must be (num_data, num_tasks)
        '''

        if self.dataset_prefix == 'ogbl':
            raise RuntimeError('ogbl link prediction dataset does not need to call save_target_labels')
    
        if not self._save_graph_list_done:
            raise RuntimeError('save_graph_list must be done beforehand.')

        if self.is_hetero:
            if not (isinstance(target_labels, dict) and len(target_labels) == 1):
                raise ValueError(f'target label must be of dictionary type with single key')

            key = list(target_labels.keys())[0]
            
            if key not in self.num_data:
                raise ValueError(f'Unknown entity type called {key}.')

            if len(target_labels[key]) != self.num_data[key]:
                raise RuntimeError(f'The length of target_labels ({len(target_labels[key])}) must be the same as the number of data points ({self.num_data[key]}).')

            if self.dataset_prefix == 'ogbg':
                raise NotImplementedError('hetero graph for graph-level prediction has not been implemented yet.')
            elif self.dataset_prefix == 'ogbn':
                np.savez_compressed(osp.join(self.raw_dir, 'node-label.npz'), **target_labels)

            self.num_tasks = target_labels[key].shape[1]


        else:
            # check type and shape
            if not isinstance(target_labels, np.ndarray):
                raise ValueError(f'target label must be of type np.ndarray')

            if len(target_labels) != self.num_data:
                raise RuntimeError(f'The length of target_labels ({len(target_labels)}) must be the same as the number of data points ({self.num_data}).')

            if self.dataset_prefix == 'ogbg':
                np.savez_compressed(osp.join(self.raw_dir, 'graph-label.npz'), graph_label = target_labels)
            elif self.dataset_prefix == 'ogbn':
                np.savez_compressed(osp.join(self.raw_dir, 'node-label.npz'), node_label = target_labels)

            self.num_tasks = target_labels.shape[1]

        self._save_target_labels_done = True

    def save_graph_list(self, graph_list):
        if not all_numpy(graph_list):
            raise RuntimeError('graph_list must only contain list/dict of numpy arrays, int, or float')

        if self.dataset_prefix == 'ogbn' or self.dataset_prefix == 'ogbl':
            if len(graph_list) > 1:
                raise RuntimeError('Multiple graphs not supported for node/link property prediction.')

        if self.is_hetero:
            self._save_graph_list_hetero(graph_list)
            self.has_node_attr = ('node_feat_dict' in graph_list[0]) and (graph_list[0]['node_feat_dict'] is not None)
            self.has_edge_attr = ('edge_feat_dict' in graph_list[0]) and (graph_list[0]['edge_feat_dict'] is not None)
        else:
            self._save_graph_list_homo(graph_list)
            self.has_node_attr = ('node_feat' in graph_list[0]) and (graph_list[0]['node_feat'] is not None)
            self.has_edge_attr = ('edge_feat' in graph_list[0]) and (graph_list[0]['edge_feat'] is not None)

        # later used for checking the shape of target_label
        if self.dataset_prefix == 'ogbg':
            self.num_data = len(graph_list) # number of graphs
        elif self.dataset_prefix == 'ogbn':
            if self.is_hetero:
                self.num_data = graph_list[0]['num_nodes_dict'] # number of nodes
            else:
                self.num_data = graph_list[0]['num_nodes'] # number of nodes
        else:
            self.num_data = None

        self._save_graph_list_done = True

    def save_split(self, split_dict, split_name):
        '''
            Save dataset split
                split_dict: must contain three keys: 'train', 'valid', 'test', where the values are the split indices stored in numpy.
                split_name (str): the name of the split
        '''

        self.split_dir = osp.join(self.dataset_dir, 'split', split_name)
        os.makedirs(self.split_dir, exist_ok=True)
        
        # verify input
        if not 'train' in split_dict:
            raise ValueError('\'train\' needs to be given in save_split')
        if not 'valid' in split_dict:
            raise ValueError('\'valid\' needs to be given in save_split')
        if not 'test' in split_dict:
            raise ValueError('\'test\' needs to be given in save_split')

        if not all_numpy(split_dict):
            raise RuntimeError('split_dict must only contain list/dict of numpy arrays, int, or float')

        ## directly save split_dict
        ## compatible with ogb>=v1.2.3
        torch.save(split_dict, osp.join(self.split_dir, 'split_dict.pt'))

        self.split_name = split_name
        self._save_split_done = True

    def copy_mapping_dir(self, mapping_dir):
        target_mapping_dir = osp.join(self.dataset_dir, 'mapping')
        os.makedirs(target_mapping_dir, exist_ok=True)
        file_list = [f for f in os.listdir(mapping_dir) if osp.isfile(osp.join(mapping_dir, f))]
        if 'README.md' not in file_list:
            raise RuntimeError(f'README.md must be included in mapping_dir {mapping_dir}')

        # copy all the files in the mapping_dir to 
        for f in file_list:
            shutil.copyfile(osp.join(mapping_dir, f), osp.join(target_mapping_dir, f))

        self._copy_mapping_dir_done = True

    def get_meta_dict(self):
        '''
            output:
                meta_dict: a dictionary that stores meta-information about data, which can be directly passed to OGB dataset object.
                Useful for debugging.
        '''

        # check everything is done before getting meta_dict
        if not self._save_graph_list_done:
            raise RuntimeError('save_graph_list not completed.')
        if not self._save_split_done:
            raise RuntimeError('save_split not completed.')
        if not self._copy_mapping_dir_done:
            raise RuntimeError('copy_mapping_dir not completed.')
        if not self._save_target_labels_done:
            raise RuntimeError('save_target_labels not completed.')
        if not self._save_task_info_done:
            raise RuntimeError('save_task_info not completed.')

        meta_dict = {'version': self.version, 'dir_path': self.dataset_dir, 'binary': 'True'}
    
        if not self.dataset_prefix == 'ogbl':
            meta_dict['num tasks'] = self.num_tasks
            meta_dict['num classes'] = self.num_classes

        meta_dict['task type'] = self.task_type

        meta_dict['eval metric'] = self.eval_metric

        meta_dict['add_inverse_edge'] = 'False'
        meta_dict['split'] = self.split_name
        meta_dict['download_name'] = self.dataset_suffix
        
        map_dict = {'ogbg': 'graphproppred', 'ogbn': 'nodeproppred', 'ogbl': 'linkproppred'}
        meta_dict['url'] = f'https://snap.stanford.edu/ogb/data/{map_dict[self.dataset_prefix]}/' + meta_dict['download_name'] + '.zip'
        meta_dict['add_inverse_edge'] = 'False'
        meta_dict['has_node_attr'] = str(self.has_node_attr)
        meta_dict['has_edge_attr'] = str(self.has_node_attr)
        meta_dict['additional node files'] = 'None'
        meta_dict['additional edge files'] = 'None'
        meta_dict['is hetero'] = str(self.is_hetero)

        # save meta-dict for submission
        torch.save(meta_dict, self.meta_dict_path)

        self._get_meta_dict_done = 'True'

        return meta_dict


    def zip(self):
        # check everything is done before zipping
        if not self._save_graph_list_done:
            raise RuntimeError('save_graph_list not completed.')
        if not self._save_split_done:
            raise RuntimeError('save_split not completed.')
        if not self._copy_mapping_dir_done:
            raise RuntimeError('copy_mapping_dir not completed.')
        if not self._save_target_labels_done:
            raise RuntimeError('save_target_labels not completed.')
        if not self._save_task_info_done:
            raise RuntimeError('save_task_info not completed.')
        if not self._get_meta_dict_done:
            raise RuntimeError('get_meta_dict not completed.')

        shutil.make_archive(self.dataset_dir, 'zip', self.dataset_dir)
        self._zip_done = True

    def cleanup(self):
        if self._zip_done:
            try:
                shutil.rmtree(self.dataset_dir)
            except FileNotFoundError:
                print('Files already deleted.')
        else:
            raise RuntimeError('Clean up after calling zip()')
                

def test_datasetsaver():
    # test on graph classification
    # ogbg-molhiv

    test_task = 'link'
    
    # testing all the dataset objects are working.
    if test_task == 'graph':
        from ogb.graphproppred import PygGraphPropPredDataset, DglGraphPropPredDataset,GraphPropPredDataset
        dataset_name = 'ogbg-molhiv'
        dataset = PygGraphPropPredDataset(dataset_name)
        dataset.get_idx_split()
        dataset = DglGraphPropPredDataset(dataset_name)
        dataset.get_idx_split()
        dataset = GraphPropPredDataset(dataset_name)
        dataset.get_idx_split()
    elif test_task == 'node':
        from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset, DglNodePropPredDataset
        dataset_name = 'ogbn-arxiv' # test ogbn-proteins
        dataset = PygNodePropPredDataset(dataset_name)
        dataset.get_idx_split()
        dataset = DglNodePropPredDataset(dataset_name)
        dataset.get_idx_split()
        dataset = NodePropPredDataset(dataset_name)
        dataset.get_idx_split()
    elif test_task == 'link':
        from ogb.linkproppred import LinkPropPredDataset, PygLinkPropPredDataset, DglLinkPropPredDataset
        dataset_name = 'ogbl-collab'
        dataset = PygLinkPropPredDataset(dataset_name)
        dataset.get_edge_split()
        dataset = DglLinkPropPredDataset(dataset_name)
        dataset.get_edge_split()
        dataset = LinkPropPredDataset(dataset_name)
        dataset.get_edge_split()
    elif test_task == 'heteronode':
        from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset, DglNodePropPredDataset
        dataset_name = 'ogbn-mag'
        dataset = PygNodePropPredDataset(dataset_name)
        dataset.get_idx_split()
        dataset = DglNodePropPredDataset(dataset_name)
        dataset.get_idx_split()
        dataset = NodePropPredDataset(dataset_name)
        dataset.get_idx_split()
    elif test_task == 'heterolink':
        from ogb.linkproppred import LinkPropPredDataset, PygLinkPropPredDataset, DglLinkPropPredDataset
        dataset_name = 'ogbl-biokg'
        dataset = PygLinkPropPredDataset(dataset_name)
        dataset.get_edge_split()
        dataset = DglLinkPropPredDataset(dataset_name)
        dataset.get_edge_split()
        dataset = LinkPropPredDataset(dataset_name)
        dataset.get_edge_split()
    else:
        raise ValueError('Invalid task category')

    print(dataset[0])
    if 'link' in test_task:
        print(dataset.get_edge_split())
    else:
        print(dataset.get_idx_split())
    
    if 'graph' in test_task:
        graph_list = dataset.graphs
    else:
        graph_list = [dataset.graph]

    if 'link' not in test_task:
        labels = dataset.labels

    is_hetero = 'hetero' in test_task
    version = 2 if dataset_name == 'ogbn-mag' else 1
    saver = DatasetSaver(dataset_name, is_hetero, version=version)

    # saving graph objects
    saver.save_graph_list(graph_list)
    # saving target labels
    if 'link' not in test_task:
        saver.save_target_labels(labels)
    # saving split
    if 'link' in test_task:
        split_idx = dataset.get_edge_split()
    else:
        split_idx = dataset.get_idx_split()
    # second argument must be the name of the split
    saver.save_split(split_idx, dataset.meta_info['split'])
    # copying mapping dir
    saver.copy_mapping_dir(f"dataset/{'_'.join(dataset_name.split('-'))}/mapping/")

    saver.save_task_info(dataset.task_type, dataset.eval_metric, dataset.num_classes if hasattr(dataset, 'num_classes') else None)

    meta_dict = saver.get_meta_dict()

    print(meta_dict)

    print('Now testing.')

    if 'graph' in test_task:
        print('library agnostic')
        dataset = GraphPropPredDataset(dataset_name, meta_dict = meta_dict)
        dataset = GraphPropPredDataset(dataset_name, meta_dict = meta_dict)
        print(dataset[0])
        print(dataset.get_idx_split())
        print('Pytorch Geometric')
        dataset = PygGraphPropPredDataset(dataset_name, meta_dict = meta_dict)
        dataset = PygGraphPropPredDataset(dataset_name, meta_dict = meta_dict)
        print(dataset[0])
        print(dataset.get_idx_split())
        print('DGL')
        dataset = DglGraphPropPredDataset(dataset_name, meta_dict = meta_dict)
        dataset = DglGraphPropPredDataset(dataset_name, meta_dict = meta_dict)
        print(dataset[0])
        print(dataset.get_idx_split())
    elif 'node' in test_task:
        print('library agnostic')
        dataset = NodePropPredDataset(dataset_name, meta_dict = meta_dict)
        dataset = NodePropPredDataset(dataset_name, meta_dict = meta_dict)
        print(dataset[0])
        print(dataset.get_idx_split())
        print('Pytorch Geometric')
        dataset = PygNodePropPredDataset(dataset_name, meta_dict = meta_dict)
        dataset = PygNodePropPredDataset(dataset_name, meta_dict = meta_dict)
        print(dataset[0])
        print(dataset.get_idx_split())
        print('DGL')
        dataset = DglNodePropPredDataset(dataset_name, meta_dict = meta_dict)
        dataset = DglNodePropPredDataset(dataset_name, meta_dict = meta_dict)
        print(dataset[0])
        print(dataset.get_idx_split())

    elif 'link' in test_task:
        print('library agnostic')
        dataset = LinkPropPredDataset(dataset_name, meta_dict = meta_dict)
        dataset = LinkPropPredDataset(dataset_name, meta_dict = meta_dict)
        print(dataset[0])
        # print(dataset.get_edge_split())
        print('Pytorch Geometric')
        dataset = PygLinkPropPredDataset(dataset_name, meta_dict = meta_dict)
        dataset = PygLinkPropPredDataset(dataset_name, meta_dict = meta_dict)
        print(dataset[0])
        # print(dataset.get_edge_split())
        print('DGL')
        dataset = DglLinkPropPredDataset(dataset_name, meta_dict = meta_dict)
        dataset = DglLinkPropPredDataset(dataset_name, meta_dict = meta_dict)
        print(dataset[0])
        # print(dataset.get_edge_split())
    else:
        raise ValueError('Invalid task category')


    # zip
    saver.zip()
    print('Finished zipping!')

    saver.cleanup()


if __name__ == '__main__':
    test_datasetsaver()

