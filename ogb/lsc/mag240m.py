from typing import Optional, Union, Dict

import os
import shutil
import os.path as osp

import torch
import numpy as np

from ogb.utils.url import decide_download, download_url, extract_zip, makedirs
from ogb.lsc.utils import split_test
from torch_geometric.data import HeteroData


class MAG240MDataset(object):
    version = 1
    # Old url hosted at Stanford
    # md5sum: bd61c9446f557fbe4430d9a7ce108b34
    # url = 'http://ogb-data.stanford.edu/data/lsc/mag240m_kddcup2021.zip'
    # New url hosted by DGL team at AWS--much faster to download
    url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/mag240m_kddcup2021.zip'

    __rels__ = {
        ('author', 'paper'): 'writes',
        ('author', 'institution'): 'affiliated_with',
        ('paper', 'paper'): 'cites',
    }

    def __init__(self, root: str = 'dataset'):
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))
        self.root = root
        self.dir = osp.join(root, 'mag240m_kddcup2021')

        if osp.isdir(self.dir) and (not osp.exists(
                osp.join(self.dir, f'RELEASE_v{self.version}.txt'))):
            print('MAG240M dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n') == 'y':
                shutil.rmtree(osp.join(self.dir))

        self.download()
        self.__meta__ = torch.load(osp.join(self.dir, 'meta.pt'))
        self.__split__ = torch.load(osp.join(self.dir, 'split_dict.pt'))

        split_test(self.__split__)

    def download(self):
        if not osp.exists(self.dir):
            if decide_download(self.url):
                path = download_url(self.url, self.root)
                extract_zip(path, self.root)
                os.unlink(path)
            else:
                print('Stop download.')
                exit(-1)

    def to_pyg_hetero_data(self):
        data = HeteroData()
        path = osp.join(self.dir, 'processed', 'paper', 'node_feat.npy')
        # Current is not in-memory
        data['paper'].x = torch.from_numpy(np.load(path, mmap_mode='r'))
        path = osp.join(self.dir, 'processed', 'paper', 'node_label.npy')
        data['paper'].y = torch.from_numpy(np.load(path))
        # torch metrics doesn't like -1
        data['paper'].y[torch.argwhere(data['paper'].y == -1)] = self.num_classes
        data.num_classes = self.num_classes
        path = osp.join(self.dir, 'processed', 'paper', 'node_year.npy')
        data['paper'].year = torch.from_numpy(np.load(path, mmap_mode='r'))

        data['author'].num_nodes = self.__meta__['author']
        # there doesnt seem to be any author data in the downloaded dir
        # ls /dfs/user/weihuahu/ogb/ogb/lsc/dataset/mag240m_kddcup2021/processed/
        # author___affiliated_with___institution  author___writes___paper  paper  paper___cites___paper
        # path = osp.join(self.dir, 'processed', 'author', 'author.npy')
        # data['author'].x = np.memmap(path, mode='r', dtype="float16", shape=(data['author'].num_nodes, self.num_paper_features))
        data['institution'].num_nodes = self.__meta__['institution']
        # again no features exist
        # path = osp.join(self.dir, 'processed', 'institution', 'inst.npy')
        # data['institution'].x = np.memmap(path, mode='r', dtype="float16", shape=(data['institution'].num_nodes, self.num_paper_features))

        for edge_type in [('author', 'affiliated_with', 'institution'),
                          ('author', 'writes', 'paper'),
                          ('paper', 'cites', 'paper')]:
            name = '___'.join(edge_type)
            path = osp.join(self.dir, 'processed', name, 'edge_index.npy')
            edge_index = torch.from_numpy(np.load(path))
            data[edge_type].edge_index = edge_index
            data[edge_type[2], f'rev_{edge_type[1]}', edge_type[0]].edge_index = edge_index.flip([0])

        for f, v in [('train', 'train'), ('valid', 'val'), ('test-dev', 'test')]:
            idx = self.get_idx_split(f)
            idx = torch.from_numpy(idx)
            mask = torch.zeros(data['paper'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['paper'][f'{v}_mask'] = mask
        return data

    @property
    def num_papers(self) -> int:
        return self.__meta__['paper']

    @property
    def num_authors(self) -> int:
        return self.__meta__['author']

    @property
    def num_institutions(self) -> int:
        return self.__meta__['institution']

    @property
    def num_paper_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return self.__meta__['num_classes']

    def get_idx_split(
        self, split: Optional[str] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        return self.__split__ if split is None else self.__split__[split]

    @property
    def paper_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_feat.npy')
        return np.load(path, mmap_mode='r')

    @property
    def all_paper_feat(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_feat.npy')
        return np.load(path)

    @property
    def paper_label(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_label.npy')
        return np.load(path, mmap_mode='r')

    @property
    def all_paper_label(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_label.npy')
        return np.load(path)

    @property
    def paper_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_year.npy')
        return np.load(path, mmap_mode='r')

    @property
    def all_paper_year(self) -> np.ndarray:
        path = osp.join(self.dir, 'processed', 'paper', 'node_year.npy')
        return np.load(path)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class MAG240MEvaluator:
    def eval(self, input_dict):
        assert 'y_pred' in input_dict and 'y_true' in input_dict

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.from_numpy(y_pred)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.from_numpy(y_true)

        assert (y_true.numel() == y_pred.numel())
        assert (y_true.dim() == y_pred.dim() == 1)

        return {'acc': int((y_true == y_pred).sum()) / y_true.numel()}

    def save_test_submission(self, input_dict: Dict, dir_path: str, mode: str):
        assert 'y_pred' in input_dict
        assert mode in ['test-whole', 'test-dev', 'test-challenge']

        y_pred = input_dict['y_pred']

        if mode == 'test-whole':
            assert y_pred.shape == (146818, )
            filename = osp.join(dir_path, 'y_pred_mag240m')
        elif mode == 'test-dev':
            assert y_pred.shape == (88092, )
            filename = osp.join(dir_path, 'y_pred_mag240m_test-dev')
        elif mode == 'test-challenge':
            assert y_pred.shape == (58726, )
            filename = osp.join(dir_path, 'y_pred_mag240m_test-challenge')

        makedirs(dir_path)
        
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        y_pred = y_pred.astype(np.short)
        np.savez_compressed(filename, y_pred=y_pred)


if __name__ == '__main__':
    dataset = MAG240MDataset()
    data = dataset.to_pyg_hetero_data()
    print(dataset)
    print(dataset.num_papers)
    print(dataset.num_authors)
    print(dataset.num_institutions)
    print(dataset.num_classes)
    split_dict = dataset.get_idx_split()
    print(split_dict['train'].shape)
    print(split_dict['valid'].shape)
    print('-----------------')
    print(split_dict['test-dev'].shape)
    print(split_dict['test-whole'].shape)
    print(split_dict['test-challenge'].shape)

    evaluator = MAG240MEvaluator()
    evaluator.save_test_submission(
        input_dict = {
        'y_pred': np.random.randint(100, size = split_dict['test-dev'].shape), 
        },
        dir_path = 'results',
        mode = 'test-dev'
    )

    evaluator.save_test_submission(
        input_dict = {
        'y_pred': np.random.randint(100, size = split_dict['test-challenge'].shape), 
        },
        dir_path = 'results',
        mode = 'test-challenge'
    )

    exit(-1)

    print(data['paper'].x.shape)
    print(data['paper'].year.shape)
    print(data['paper'].year[:100])
    print(data[(('author', 'writes', 'paper'))].edge_index.shape)
    print(data[('author', 'affiliated_with', 'institution')].edge_index.shape)
    print(data[('paper', 'cites', 'paper')].edge_index.shape)
    print(data[('author', 'writes', 'paper')].edge_index[:, :10])
    print(data[('author', 'affiliated_with', 'institution')].edge_index[:, :10])
    print(data[('paper', 'cites', 'paper')].edge_index[:, :10])
    print('-----------------')

    train_idx = dataset.get_idx_split('train')
    val_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test-whole')
    print(len(train_idx) + len(val_idx) + len(test_idx))

    print(data['paper'].y[train_idx][:10])
    print(data['paper'].y[val_idx][:10])
    print(data['paper'].y[test_idx][:10])
    print(data['paper'].year[train_idx][:10])
    print(data['paper'].year[val_idx][:10])
    print(data['paper'].year[test_idx][:10])
