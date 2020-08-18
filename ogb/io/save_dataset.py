import torch
import pandas as pd
import os
from datetime import date
import shutil
from tqdm import tqdm
import numpy as np
from ogb.io.read_graph_raw import read_csv_graph_raw
from ogb.utils.torch_util import all_numpy

class DatasetSaver(object):
    '''
        A class for saving graphs and split in OGB-compatible manner
    '''
    def __init__(self, dataset_name, is_hetero, version):
        # verify input
        if not ('ogbn-' in dataset_name or 'ogbl-' in dataset_name or 'ogbg-' in dataset_name):
            raise ValueError('Dataset name must have valid ogb prefix (e.g., ogbn-*).')
        if not isinstance(is_hetero, bool):
            raise ValueError('is_hetero must be of type bool.')
        if not (isinstance(version, int) and version >= 0):
            raise ValueError('version must be of type int and non-negative')

        self.dataset_name = dataset_name

        self.is_hetero = is_hetero

        self.dataset_dir = '_'.join(dataset_name.split('-')[1:])
        self.dataset_prefix = dataset_name.split('-')[0] # specify the task category

        if os.path.exists(self.dataset_dir):
            if input(f"Found an existing directory at {self.dataset_dir}/. \nWill you remove it? (y/N)\n").lower() == "y":
                shutil.rmtree(self.dataset_dir)
                print('Removed existing directory')
            else:
                print('Process stopped.')
                exit(-1)


        # make necessary dirs
        self.raw_dir = os.path.join(self.dataset_dir, 'raw')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, 'processed'), exist_ok=True)

        # create release note
        with open(os.path.join(self.dataset_dir, f'RELEASE_v{version}.txt'), 'w') as fw:
            fw.write(f'# Release note for {self.dataset_name}\n\n### v{version}: {date.today()}')

        # check list
        self._save_graph_list_done = False
        self._save_split_done = False
        self._copy_mapping_dir_done = False

        if 'ogbl' == self.dataset_prefix:
            self._save_target_label_done = True # for ogbl, we do not need to give predicted labels
        else:
            self._save_target_label_done = False # for ogbn and ogbg, need to give predicted labels
        
        self._zip_done = False

    def _save_graph_list_hetero(self, graph_list):
        pass 

    def _save_graph_list_homo(self, graph_list):
        if self.dataset_prefix == 'ogbn' or self.dataset_prefix == 'ogbl':
            if len(graph_list) > 1:
                raise RuntimeError('Multiple graphs not supported for node/link property prediction.')
        
        dict_keys = graph_list[0].keys()
        # check necessary keys
        if not 'edge_index' in dict_keys:
            raise RuntimeError('edge_index needs to be provided in graph objects')
        if not 'num_nodes' in dict_keys:
            raise RuntimeError('num_nodes needs to be provided in graph objects')

        print(dict_keys)

        # saving edge_index
        print('Saving edge_index')
        edge = np.concatenate([graph['edge_index'] for graph in graph_list], axis = 1).transpose().astype(np.int64)
        num_edges_list = np.array([graph['edge_index'].shape[1] for graph in graph_list]).astype(np.int64)

        if edge.shape[1] != 2:
            raise RuntimeError('edge_index must have shape (2, num_edges)')

        pd.DataFrame(edge).to_csv(os.path.join(self.raw_dir, 'edge.csv.gz'), index=False, compression="gzip", header=False)
        pd.DataFrame(num_edges_list).to_csv(os.path.join(self.raw_dir, 'num-edge-list.csv.gz'), index=False, compression="gzip", header=False)

        # saving num_nodes
        print('Saving num_nodes')
        num_nodes_list = np.array([graph['num_nodes'] for graph in graph_list]).astype(np.int64)
        pd.DataFrame(num_nodes_list).to_csv(os.path.join(self.raw_dir, 'num-node-list.csv.gz'), index=False, compression="gzip", header=False)

        additional_node_files = []
        additional_edge_files = []

        for key in dict_keys:
            if key == 'edge_index' or key == 'num_nodes':
                continue 
            if graph_list[0][key] is None:
                continue

            print(f'Saving {key}')

            if 'node_' in key:
                # make sure saved in np.int64 or np.float32
                dtype = np.int64 if 'int' in str(graph_list[0][key].dtype) else np.float32
                # check num_nodes
                for i in range(len(graph_list)):
                    if len(graph_list[i][key]) != num_nodes_list[i]:
                        raise RuntimeError(f'num_nodes mistmatches with {key}')

                cat_feat = np.concatenate([graph[key] for graph in graph_list], axis = 0).astype(dtype)
                if key == 'node_feat':
                    pd.DataFrame(cat_feat).to_csv(os.path.join(self.raw_dir, 'node-feat.csv.gz'), index=False, compression="gzip", header=False)
                else:
                    additional_node_files.append(key)
                    pd.DataFrame(cat_feat).to_csv(os.path.join(self.raw_dir, f'{key}.csv.gz'), index=False, compression="gzip", header=False)

            elif 'edge_' in key:
                # make sure saved in np.int64 or np.float32
                dtype = np.int64 if 'int' in str(graph_list[0][key].dtype) else np.float32
                # check num_edges
                for i in range(len(graph_list)):
                    if len(graph_list[i][key]) != num_edges_list[i]:
                        raise RuntimeError(f'num_edges mistmatches with {key}')

                cat_feat = np.concatenate([graph[key] for graph in graph_list], axis = 0).astype(dtype)
                if key == 'edge_feat':
                    pd.DataFrame(cat_feat).to_csv(os.path.join(self.raw_dir, 'edge-feat.csv.gz'), index=False, compression="gzip", header=False)
                else:
                    additional_edge_files.append(key)
                    pd.DataFrame(cat_feat).to_csv(os.path.join(self.raw_dir, f'{key}.csv.gz'), index=False, compression="gzip", header=False)

            else:
                raise RuntimeError(f'Keys in graph object should start from either \'node_\' or \'edge_\', but \'{key}\' given.')

        print('Finished saving all the files!')
        print('Validating...')
        # testing
        print(f'Additional node files: {additional_node_files}')
        print(f'Additional edge files: {additional_edge_files}')
        print('Reading saved files')
        graph_list_read = read_csv_graph_raw(self.raw_dir, False, additional_node_files, additional_edge_files)

        print('Checking read graphs and given graphs are the same')
        for i in tqdm(range(len(graph_list))):
            # assert(graph_list[i].keys() == graph_list_read[i].keys())
            for key in graph_list[i].keys():
                if graph_list[i][key] is not None:
                    equal = graph_list[i][key] == graph_list_read[i][key]
                    if isinstance(equal, np.ndarray):
                        assert((graph_list[i][key] == graph_list_read[i][key]).all())
                    else:
                        assert(equal)

        del graph_list_read

    def save_target_label(self, target_label):
        if not isinstance(target_label, np.ndarray):
            raise ValueError(f"target label must be of type np.ndarray")
            
        if self.dataset_prefix == 'ogbg':
            pd.DataFrame(target_label).to_csv(os.path.join(self.raw_dir, 'graph-label.csv.gz'), index=False, compression="gzip", header=False)
        elif self.dataset_prefix == 'ogbn':
            pd.DataFrame(target_label).to_csv(os.path.join(self.raw_dir, 'node-label.csv.gz'), index=False, compression="gzip", header=False)
        elif self.dataset_prefix == 'ogbl':
            raise RuntimeError('Target labels do not need to be saved for link prediction dataset')

        self._save_target_label_done = True

    def save_graph_list(self, graph_list):
        if not all_numpy(graph_list):
            raise RuntimeError('graph_list must only contain list/dict of numpy arrays, int, or float')

        if self.is_hetero:
            self._save_graph_list_hetero(graph_list)
        else:
            # (TODO) write this
            self._save_graph_list_homo(graph_list)

        self._save_graph_list_done = True


    def save_split(self, split_dict, split_name):
        self.split_dir = os.path.join(self.dataset_dir, 'split', split_name)
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
        torch.save(split_dict, os.path.join(self.split_dir, 'split_dict.pt'))

        self._save_split_done = True

    def copy_mapping_dir(self, mapping_dir):
        target_mapping_dir = os.path.join(self.dataset_dir, 'mapping')
        os.makedirs(target_mapping_dir, exist_ok=True)
        file_list = [f for f in os.listdir(mapping_dir) if os.path.isfile(os.path.join(mapping_dir, f))]
        if 'README.md' not in file_list:
            raise RuntimeError(f'README.md must be included in mapping_dir {mapping_dir}')

        # copy all the files in the mapping_dir to 
        for f in file_list:
            shutil.copyfile(os.path.join(mapping_dir, f), os.path.join(target_mapping_dir, f))

        self._copy_mapping_dir_done = True

    def zip(self):
        if not self._save_graph_list_done:
            raise RuntimeError('save_graph_list not completed.')
        if not self._save_split_done:
            raise RuntimeError('save_split not completed.')
        if not self._copy_mapping_dir_done:
            raise RuntimeError('copy_mapping_dir not completed.')
        if not self._save_target_label_done:
            raise RuntimeError('save_target_label not completed.')

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
    
    if test_task == 'graph':
        from ogb.graphproppred import GraphPropPredDataset
        dataset_name = 'ogbg-molhiv'
        dataset = GraphPropPredDataset(dataset_name)
    elif test_task == 'node':
        from ogb.nodeproppred import NodePropPredDataset
        dataset_name = 'ogbn-arxiv'
        dataset = NodePropPredDataset(dataset_name)
    elif test_task == 'link':
        from ogb.linkproppred import LinkPropPredDataset
        dataset_name = 'ogbl-collab'
        dataset = LinkPropPredDataset(dataset_name)
    elif test_task == 'heteronode':
        from ogb.nodeproppred import NodePropPredDataset
        dataset_name = 'ogbn-mag'
        dataset = NodePropPredDataset(dataset_name)
    else:
        raise ValueError('Invalid task category')

    is_hetero = 'hetero' in test_task
    print(dataset[0])
    if test_task == 'link':
        print(dataset.get_edge_split())
    else:
        print(dataset.get_idx_split())
    
    if test_task == 'graph':
        graph_list = dataset.graphs
    else:
        graph_list = [dataset.graph]

    if test_task != 'link':
        label_list = dataset.labels

    saver = DatasetSaver(dataset_name, is_hetero, version=1)
    # saving graph objects
    saver.save_graph_list(graph_list)
    # saving target labels
    if test_task != 'link':
        saver.save_target_label(label_list)
    # saving split
    if test_task == 'link':
        split_idx = dataset.get_edge_split()
    else:
        split_idx = dataset.get_idx_split()
    # second argument must be the name of the split
    saver.save_split(split_idx, dataset.meta_info[dataset_name]["split"])
    # copying mapping dir
    saver.copy_mapping_dir(f'dataset/{"_".join(dataset_name.split("-"))}/mapping/')
    # zip
    saver.zip()

    print('Finished zipping!')
    print('Now testing.')

    if test_task == 'graph':
        dataset = GraphPropPredDataset(dataset_name, dir_path = saver.dataset_dir)
    elif test_task == 'node':
        dataset = NodePropPredDataset(dataset_name, dir_path = saver.dataset_dir)
    elif test_task == 'link':
        dataset = LinkPropPredDataset(dataset_name, dir_path = saver.dataset_dir)
    else:
        raise ValueError('Invalid task category')

    print(dataset[0])
    if test_task == 'link':
        print(dataset.get_edge_split())
    else:
        print(dataset.get_idx_split())

    saver.cleanup()


if __name__ == '__main__':
    test_datasetsaver()

