from typing import Optional, Union, Dict

import os
import os.path as osp
import shutil

from ogb.utils import smiles2graph
from ogb.utils.url import decide_download, download_url, extract_zip

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

class PCQM4Mv2Dataset(object):
    def __init__(self, root = 'dataset', smiles2graph = smiles2graph, only_smiles=False):
        '''
        Library-agnostic PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
            - only_smiles (bool): If this is true, we directly return the SMILES string in our __get_item__, without converting it into a graph.
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.only_smiles = only_smiles
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PCQM4Mv2Dataset, self).__init__()

        # Prepare everything.
        # download if there is no raw file
        # preprocess if there is no processed file
        # load data if processed file is found.
        if self.only_smiles:
            self.prepare_smiles()
        else:
            self.prepare_graph()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def prepare_smiles(self):
        raw_dir = osp.join(self.folder, 'raw')
        if not osp.exists(osp.join(raw_dir, 'data.csv.gz')):
            # if the raw file does not exist, then download it.
            self.download()

        data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles'].values
        homolumogap_list = data_df['homolumogap'].values
        self.graphs = list(smiles_list)
        self.labels = homolumogap_list

    def prepare_graph(self):
        processed_dir = osp.join(self.folder, 'processed')
        raw_dir = osp.join(self.folder, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'data_processed')

        if osp.exists(pre_processed_file_path):        
            # if pre-processed file already exists
            loaded_dict = torch.load(pre_processed_file_path, 'rb')
            self.graphs, self.labels = loaded_dict['graphs'], loaded_dict['labels']
        
        else:
            # if pre-processed file does not exist
            
            if not osp.exists(osp.join(raw_dir, 'data.csv.gz')):
                # if the raw file does not exist, then download it.
                self.download()

            data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
            smiles_list = data_df['smiles']
            homolumogap_list = data_df['homolumogap']

            print('Converting SMILES strings into graphs...')
            self.graphs = []
            self.labels = []
            for i in tqdm(range(len(smiles_list))):

                smiles = smiles_list[i]
                homolumogap = homolumogap_list[i]
                graph = self.smiles2graph(smiles)
                
                assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert(len(graph['node_feat']) == graph['num_nodes'])

                self.graphs.append(graph)
                self.labels.append(homolumogap)

            self.labels = np.array(self.labels)
            print(self.labels)

            # double-check prediction target
            split_dict = self.get_idx_split()
            assert(all([not np.isnan(self.labels[i]) for i in split_dict['train']]))
            assert(all([not np.isnan(self.labels[i]) for i in split_dict['valid']]))
            assert(all([np.isnan(self.labels[i]) for i in split_dict['test-dev']]))
            assert(all([np.isnan(self.labels[i]) for i in split_dict['test-challenge']]))

            print('Saving...')
            torch.save({'graphs': self.graphs, 'labels': self.labels}, pre_processed_file_path, pickle_protocol=4)

    def get_idx_split(self):
        split_dict = torch.load(osp.join(self.folder, 'split_dict.pt'))
        return split_dict

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


class PCQM4Mv2Evaluator:
    def __init__(self):
        '''
            Evaluator for the PCQM4Mv2 dataset
            Metric is Mean Absolute Error
        '''
        pass 

    def eval(self, input_dict):
        '''
            y_true: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_pred: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_true and y_pred need to be of the same type (either numpy.ndarray or torch.Tensor)
        '''
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)
        assert(len(y_true.shape) == 1)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}

    def save_test_submission(self, input_dict: Dict, dir_path: str, mode: str):
        '''
            save test submission file at dir_path
        '''
        assert('y_pred' in input_dict)
        assert mode in ['test-dev', 'test-challenge']

        y_pred = input_dict['y_pred']

        if mode == 'test-dev':
            filename = osp.join(dir_path, 'y_pred_pcqm4m-v2_test-dev')
            assert(y_pred.shape == (147037,))
        elif mode == 'test-challenge':
            filename = osp.join(dir_path, 'y_pred_pcqm4m-v2_test-challenge')
            assert(y_pred.shape == (147432,))

        assert(isinstance(filename, str))
        assert(isinstance(y_pred, np.ndarray) or isinstance(y_pred, torch.Tensor))

        if not osp.exists(dir_path):
            os.makedirs(dir_path)

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred = y_pred.astype(np.float32)
        np.savez_compressed(filename, y_pred = y_pred)

if __name__ == '__main__':
    dataset = PCQM4Mv2Dataset(only_smiles=True)
    print(dataset)
    print(dataset[1234])

    split_dict = dataset.get_idx_split()
    print(split_dict['train'].shape)
    print(split_dict['valid'].shape)
    print('-----------------')
    print(split_dict['test-dev'].shape)
    print(split_dict['test-challenge'].shape)


    evaluator = PCQM4Mv2Evaluator()
    y_true = torch.randn(100)
    y_pred = torch.randn(100)
    result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
    print(result)

    print(len(split_dict['test-dev']))
    print(len(split_dict['test-challenge']))

    y_pred = torch.randn(len(split_dict['test-dev']))
    evaluator.save_test_submission({'y_pred': y_pred}, 'results',mode = 'test-dev')

    y_pred = torch.randn(len(split_dict['test-challenge']))
    evaluator.save_test_submission({'y_pred': y_pred}, 'results',mode = 'test-challenge')


    

        

        

