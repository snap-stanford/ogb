import pandas as pd
import shutil, os
import numpy as np
import os.path as osp
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_raw import read_csv_graph_raw, read_binary_graph_raw
import torch

class GraphPropPredDataset(object):
    def __init__(self, name, root = 'dataset', meta_dict = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name ## original name, e.g., ogbg-hib
        
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) ## replace hyphen with underline, e.g., ogbg_hiv
            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user. 
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.num_classes = self.meta_info['num classes']
        self.binary = self.meta_info['binary'] == 'True'

        super(GraphPropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        raw_dir = osp.join(self.root, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'data_processed')

        if os.path.exists(pre_processed_file_path):
            loaded_dict = torch.load(pre_processed_file_path, 'rb')
            self.graphs, self.labels = loaded_dict['graphs'], loaded_dict['labels']

        else:
            ### check download
            if self.binary:
                # npz format
                has_necessary_file = osp.exists(osp.join(self.root, 'raw', 'data.npz'))
            else:
                # csv file
                has_necessary_file = osp.exists(osp.join(self.root, 'raw', 'edge.csv.gz'))
            
            ### download
            if not has_necessary_file:
                url = self.meta_info['url']
                if decide_download(url):
                    path = download_url(url, self.original_root)
                    extract_zip(path, self.original_root)
                    os.unlink(path)
                    # delete folder if there exists
                    try:
                        shutil.rmtree(self.root)
                    except:
                        pass
                    shutil.move(osp.join(self.original_root, self.download_name), self.root)
                else:
                    print('Stop download.')
                    exit(-1)

            ### preprocess
            add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

            if self.meta_info['additional node files'] == 'None':
                additional_node_files = []
            else:
                additional_node_files = self.meta_info['additional node files'].split(',')

            if self.meta_info['additional edge files'] == 'None':
                additional_edge_files = []
            else:
                additional_edge_files = self.meta_info['additional edge files'].split(',')
            
            if self.binary:
                self.graphs = read_binary_graph_raw(raw_dir, add_inverse_edge = add_inverse_edge)
            else:
                self.graphs = read_csv_graph_raw(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)

            if self.task_type == 'subtoken prediction':
                labels_joined = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
                # need to split each element into subtokens
                self.labels = [str(labels_joined[i][0]).split(' ') for i in range(len(labels_joined))]
            else:
                if self.binary:
                    self.labels = np.load(osp.join(raw_dir, 'graph-label.npz'))['graph_label']
                else:
                    self.labels = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values

            print('Saving...')
            torch.save({'graphs': self.graphs, 'labels': self.labels}, pre_processed_file_path, pickle_protocol=4)


    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']
            
        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, (int, np.integer)):
            return self.graphs[idx], self.labels[idx]

        raise IndexError(
            'Only integer is valid index (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


if __name__ == '__main__':
    dataset = GraphPropPredDataset(name = 'ogbg-code')
    # target_list = np.array([len(label) for label in dataset.labels])
    # print(np.sum(target_list == 1)/ float(len(target_list)))
    # print(np.sum(target_list == 2)/ float(len(target_list)))
    # print(np.sum(target_list == 3)/ float(len(target_list)))

    # from collections import Counter
    # print(Counter(target_list))

    print(dataset.num_classes)
    split_index = dataset.get_idx_split()
    print(split_index)
    # print(dataset)
    # print(dataset[2])
    # print(split_index['train'])
    # print(split_index['valid'])
    # print(split_index['test'])


