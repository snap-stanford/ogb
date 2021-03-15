from typing import Optional, Union, Dict

import os
import shutil
import os.path as osp

import torch
import numpy as np

from ogb.utils.url import decide_download, download_url, extract_zip, makedirs


class WikiKG90MDataset(object):

    def __init__(self, root: str = 'dataset'):
        self.original_root = root

        self.folder = osp.join(root, 'wikikg90m_kddcup2021')
        self.version = 1
        self.url = f'http://ogb-data.stanford.edu/data/lsc/wikikg90m_kddcup2021.zip'
        self.processed_dir = osp.join(self.folder, 'processed')

        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('WikiKG90M dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n') == 'y':
                shutil.rmtree(osp.join(self.folder))

        self.download()
        self.__meta__ = torch.load(osp.join(self.folder, 'meta.pt'))

        # training triplet
        path = osp.join(self.processed_dir, 'train_hrt.npy')
        self._train_hrt = np.load(path)

        # node/edge features
        self._entity_feat = None
        self._all_entity_feat = None
        self._relation_feat = None

        # Validation
        self._valid_dict = None

        # Test
        self._test_dict = None

    def download(self):
        if not osp.exists(self.folder):
            if decide_download(self.url):
                path = download_url(self.url, self.original_root)
                extract_zip(path, self.original_root)
                os.unlink(path)
            else:
                print('Stop download.')
                exit(-1)

    @property
    def num_entities(self) -> int:
        return self.__meta__['num_entities']

    @property
    def num_relations(self) -> int:
        return self.__meta__['num_relations']

    @property
    def num_feat_dims(self) -> int:
        '''
            Dimensionality of relation and entity features obtained by roberta
        '''
        return 768

    @property
    def entity_feat(self) -> np.ndarray:
        '''
            Entity feature
            - np.ndarray of shape (num_entities, num_feat_dims)
              i-th row stores the feature of i-th entity
              * Loading everything into memory at once
              * saved in np.float16
        '''
        if self._entity_feat is None:
            path = osp.join(self.processed_dir, 'entity_feat.npy')
            self._entity_feat = np.load(path, mmap_mode='r')
        return self._entity_feat

    @property
    def all_entity_feat(self) -> np.ndarray:
        if self._all_entity_feat is None:
            path = osp.join(self.processed_dir, 'entity_feat.npy')
            self._all_entity_feat = np.load(path)
        return self._all_entity_feat

    @property
    def relation_feat(self) -> np.ndarray:
        '''
            Relation feature
            - np.ndarray of shape (num_relations, num_feat_dims)
              i-th row stores the feature of i-th relation
              * saved in np.float16
        '''
        if self._relation_feat is None:
            path = osp.join(self.processed_dir, 'relation_feat.npy')
            self._relation_feat = np.load(path)
        return self._relation_feat

    @property
    def all_relation_feat(self) -> np.ndarray:
        '''
            For completeness.
            #relations is small, so everything can be loaded into CPU memory.
        '''
        return self.relation_feat

    @property
    def train_hrt(self) -> np.ndarray:
        '''
            Training triplets (h, r, t)
            - np.ndarray of shape (num_triplets, 3)
            - i-th row corresponds to the i-th triplet (h, r, t)
        '''
        return self._train_hrt

    @property
    def valid_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        '''
            - h,r->t: Given head and relation, predict target entities
                - hr: np.ndarray of shape (num_validation_triplets, 2)
                      i-th row stores i-th (h,r)
                - t_candidate: np.ndarray of shape (num_validation_triplets, 1001)
                      i-th row stores i-th candidates for the tail entities
                      * Using mmap_mode
                - t_correct_index: np.ndarray of shape (num_validation_triplets,)
                      i-th row stores the index (0<= index < 1001) of the true tail entity in t_candidate[i]
                      i.e., (h[i],r[i],t_candidate[i][t_correct_index[i]]) is the true triplet.
        '''
        if self._valid_dict is None:
            self._valid_dict = {}
            # h, r -> t
            self._valid_dict['h,r->t'] = {}
            self._valid_dict['h,r->t']['hr'] = np.load(osp.join(self.processed_dir, 'val_hr.npy'))
            self._valid_dict['h,r->t']['t_candidate'] = np.load(osp.join(self.processed_dir, 'val_t_candidate.npy'), mmap_mode='r')
            self._valid_dict['h,r->t']['t_correct_index'] = np.load(osp.join(self.processed_dir, 'val_t_correct_index.npy'))
        return self._valid_dict

    @property
    def test_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        '''
            - h,r->t: Given head and relation, predict target entities
                - hr: np.ndarray of shape (num_test_triplets, 2)
                      i-th row stores i-th (h,r)
                - t_candidate: np.ndarray of shape (num_test_triplets, 1001)
                      i-th row stores i-th candidates for the tail entities
                      * Using mmap_mode
        '''
        if self._test_dict is None:
            self._test_dict = {}
            # h, r -> t
            self._test_dict['h,r->t'] = {}
            self._test_dict['h,r->t']['hr'] = np.load(osp.join(self.processed_dir, 'test_hr.npy'))
            self._test_dict['h,r->t']['t_candidate'] = np.load(osp.join(self.processed_dir, 'test_t_candidate.npy'), mmap_mode='r')

        return self._test_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class WikiKG90MEvaluator:
    def eval(self, input_dict):
        '''
            Format of input_dict:
            - 'h,r->t'
                - t_pred_top10: np.ndarray of shape (num_eval_triplets, 10)
                    each element must < 1001
                    (i,j) represents the j-th prediction for i-th triplet
                    Only top10 prediction is taken into account
                - t_correct_index: np.ndarray of shape (num_eval_triplets,)

        '''
        assert 'h,r->t' in input_dict
        assert ('t_pred_top10' in input_dict['h,r->t']) and ('t_correct_index' in input_dict['h,r->t'])

        # h,r->t
        t_pred_top10 = input_dict['h,r->t']['t_pred_top10']
        t_correct_index = input_dict['h,r->t']['t_correct_index']
        if not isinstance(t_pred_top10, torch.Tensor):
            t_pred_top10 = torch.from_numpy(t_pred_top10)
        if not isinstance(t_correct_index, torch.Tensor):
            t_correct_index = torch.from_numpy(t_correct_index)

        assert t_pred_top10.shape[1] == 10 and len(t_pred_top10) == len(t_correct_index)
        assert (0 <= t_pred_top10).all() and (t_pred_top10 < 1001).all()
        assert (0 <= t_correct_index).all() and (t_correct_index < 1001).all()

        mrr = self._calculate_mrr(t_correct_index.to(t_pred_top10.device), t_pred_top10)

        return {'mrr': mrr}

    def _calculate_mrr(self, correct_index, pred_top10):
        '''
            - correct_index: shape (num_eval_triplets, )
            - pred_top10: shape (num_eval_triplets, 10)
        '''
        # extract indices where correct_index is within top10
        tmp = torch.nonzero(correct_index.view(-1,1) == pred_top10)

        # reciprocal rank
        # if rank is larger than 10, then set the reciprocal rank to 0.
        rr = torch.zeros(len(correct_index)).to(tmp.device)
        rr[tmp[:,0]] = 1./(tmp[:,1].float() + 1.)

        # mean reciprocal rank
        return float(rr.mean().item())

    def save_test_submission(self, input_dict, dir_path):
        assert 'h,r->t' in input_dict
        assert 't_pred_top10' in input_dict['h,r->t']

        t_pred_top10 = input_dict['h,r->t']['t_pred_top10']

        assert t_pred_top10.shape == (1359303, 10) and (0 <= t_pred_top10).all() and (t_pred_top10 < 1001).all()

        if isinstance(t_pred_top10, torch.Tensor):
            t_pred_top10 = t_pred_top10.cpu().numpy()
        t_pred_top10 = t_pred_top10.astype(np.int16)

        makedirs(dir_path)
        filename = osp.join(dir_path, 't_pred_wikikg90m')
        np.savez_compressed(filename, t_pred_top10=t_pred_top10)

if __name__ == '__main__':
    dataset = WikiKG90MDataset()
    print(dataset)
    print(dataset.num_entities)
    print(dataset.entity_feat)
    print(dataset.entity_feat.shape)
    print(dataset.num_relations)
    print(dataset.relation_feat)
    print(dataset.all_relation_feat)
    print(dataset.relation_feat.shape)
    print(dataset.train_hrt)
    print(dataset.valid_dict)
    print(dataset.test_dict)
    print(dataset.valid_dict['h,r->t']['t_correct_index'].max())
    print(dataset.valid_dict['h,r->t']['t_correct_index'].min())

    evaluator = WikiKG90MEvaluator()

    valid_dict = dataset.valid_dict
    t_correct_index = valid_dict['h,r->t']['t_correct_index']
    test_task = dataset.test_dict['h,r->t']
    
    # t_correct_index = test_task['t_correct_index'] # key error
    
    hr = test_task['hr']
    t_candidate = test_task['t_candidate']
    t_pred_top10 = np.random.randint(0,1001, size=(len(t_correct_index), 10))

    input_dict = {}
    input_dict['h,r->t'] = {'t_correct_index': t_correct_index, 't_pred_top10': t_pred_top10}
    result = evaluator.eval(input_dict)
    print(result)

    t_correct_index = torch.from_numpy(t_correct_index)
    t_pred_top10 = torch.from_numpy(t_pred_top10)

    input_dict = {}
    input_dict['h,r->t'] = {'t_correct_index': t_correct_index, 't_pred_top10': t_pred_top10}
    result = evaluator.eval(input_dict)
    print(result)

    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': np.random.randint(0,1001, size = (1359303, 10))}
    evaluator.save_test_submission(input_dict, 'result')

    # print(dataset.all_entity_feat)
    # print(dataset.all_entity_feat.shape)
