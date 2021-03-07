from typing import Optional, Union, Dict

import os
import shutil
import os.path as osp

import torch
import numpy as np

from ogb.utils.url import decide_download, download_url, extract_zip, makedirs


class MAG240MDataset(object):
    version = 1
    url = 'http://ogb-data.stanford.edu/data/lsc/mag240m_kddcup2021.zip'

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

    def download(self):
        if not osp.exists(self.dir):
            if decide_download(self.url):
                path = download_url(self.url, self.root)
                extract_zip(path, self.root)
                os.unlink(path)
            else:
                print('Stop download.')
                exit(-1)

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

    def edge_index(self, id1: str, id2: str,
                   id3: Optional[str] = None) -> np.ndarray:
        src = id1
        rel, dst = (id3, id2) if id3 is None else (id2, id3)
        rel = self.__rels__[(src, dst)] if rel is None else rel
        name = f'{src}___{rel}___{dst}'
        path = osp.join(self.dir, 'processed', name, 'edge_index.npy')
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

    def save_test_submission(self, input_dict, dir_path):
        assert 'y_pred' in input_dict
        y_pred = input_dict['y_pred']
        assert y_pred.shape == (146818, )

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        y_pred = y_pred.astype(np.short)

        makedirs(dir_path)
        filename = osp.join(dir_path, 'y_pred_mag240m')
        np.savez_compressed(filename, y_pred=y_pred)


if __name__ == '__main__':
    dataset = MAG240MDataset()
    print(dataset)
    print(dataset.num_papers)
    print(dataset.num_authors)
    print(dataset.num_institutions)
    print(dataset.num_classes)
    split_dict = dataset.get_idx_split()
    print(split_dict['train'].shape)
    print(split_dict['valid'].shape)
    print('-----------------')
    print(split_dict['test'].shape)

    print(dataset.paper_feat.shape)
    print(dataset.paper_year.shape)
    print(dataset.paper_year[:100])
    print(dataset.edge_index('author', 'paper').shape)
    print(dataset.edge_index('author', 'writes', 'paper').shape)
    print(dataset.edge_index('author', 'writes', 'paper')[:, :10])
    print('-----------------')

    train_idx = dataset.get_idx_split('train')
    val_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')
    print(len(train_idx) + len(val_idx) + len(test_idx))
    print(dataset.paper_label[train_idx][:10])
    print(dataset.paper_label[val_idx][:10])
    print(dataset.paper_label[test_idx][:10])
    print(dataset.paper_year[train_idx][:10])
    print(dataset.paper_year[val_idx][:10])
    print(dataset.paper_year[test_idx][:10])
