from typing import Optional

import os
import shutil
import os.path as osp

import torch
import numpy as np

from ogb.utils.url import decide_download, download_url, extract_zip


class MAG240mDataset(object):
    url = 'https://snap.stanford.edu/ogb/data/lsc/mag240m.zip'
    version = 0

    def __init__(self, root: str = 'dataset'):
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))
        self.root = root
        self.dir = osp.join(root, 'mag240m')

        if not osp.exists(osp.join(self.dir, f'RELEASE_v{self.version}.txt')):
            print('MAG240m dataset has been updated.')
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

    def get_idx_split(self, split: Optional[str] = None):
        return self.__split__ if split is None else self.__split__[split]

    @property
    def paper_feat(self):
        path = osp.join(self.dir, 'processed', 'paper', 'node_feat.npy')
        return np.load(path, mmap_mode='r')

    @property
    def paper_label(self):
        path = osp.join(self.dir, 'processed', 'paper', 'node_label.npy')
        return np.load(path, mmap_mode='r')

    @property
    def paper_year(self):
        path = osp.join(self.dir, 'processed', 'paper', 'node_year.npy')
        return np.load(path, mmap_mode='r')

    def edge_index(self, src: str, rel: str, dst: str):
        name = f'{src}___{rel}___{dst}'
        path = osp.join(self.dir, 'processed', name, 'edge_index.npy')
        return np.load(path, mmap_mode='r')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


if __name__ == '__main__':
    dataset = MAG240mDataset('/data/datasets/OGB/mag240m')
    print(dataset)
    print(dataset.num_papers)
    print(dataset.num_authors)
    print(dataset.num_institutions)
    print(dataset.num_classes)
    split_dict = dataset.get_idx_split()
    print(split_dict['train'].shape)
    print(split_dict['valid'].shape)
    print(split_dict['test'].shape)

    print(dataset.paper_feat.shape)
    print(dataset.paper_year.shape)
    print(dataset.paper_year[:100])
    print(dataset.edge_index('author', 'writes', 'paper').shape)
    print(dataset.edge_index('author', 'writes', 'paper')[:, :10])
