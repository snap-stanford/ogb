from typing import Optional, Union, Dict

import os
import shutil
import os.path as osp

import torch
import numpy as np

from ogb.utils.url import decide_download, download_url, extract_zip, makedirs


class WikiKG90MDataset(object):
    version = 0
    url = 'https://snap.stanford.edu/ogb/data/lsc/wikikg90m-folder.zip'

    def __init__(self, root: str = 'dataset'):
        self.original_root = root
        self.folder = osp.join(root, 'wikikg90m-folder')
        self.download_name = 'wikikg90m-folder'
        self.version = 1
        self.url = f'http://ogb-data.stanford.edu/data/lsc/{self.download_name}.zip'
        self.processed_dir = osp.join(self.folder, 'processed')

        # if not osp.exists(osp.join(self.dir, f'RELEASE_v{self.version}.txt')):
        #     print('WikiKG90M dataset has been updated.')
        #     if input('Will you update the dataset now? (y/N)\n') == 'y':
        #         shutil.rmtree(osp.join(self.dir))

        # self.download()
        self.__meta__ = torch.load(osp.join(self.folder, 'meta.pt'))

        # training triplet
        path = osp.join(self.processed_dir, 'train_hrt.npy')
        self._train_hrt = np.load(path)

        # node/edge features
        self._entity_feat = None
        self._relation_feat = None

        # Validation
        self._valid_dict = None

        # Test
        self._test_dict = None

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            try:
                shutil.rmtree(self.folder)
            except:
                pass
            shutil.move(osp.join(self.original_root, self.download_name), self.folder)
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
              * Using mmap_mode
              * saved in np.float16
        '''
        if self._entity_feat is None:
            path = osp.join(self.processed_dir, 'entity_feat.npy')
            self._entity_feat = np.load(path, mmap_mode='r')
        return self._entity_feat

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
            Two validation tasks: h,r->t and t,r->h

            - h,r->t: Given head and relation, predict target entities
                - hr: np.ndarray of shape (num_validation_triplets, 2)
                      i-th row stores i-th (h,r)
                - t_candidate: np.ndarray of shape (num_validation_triplets, 1001)
                      i-th row stores i-th candidates for the tail entities
                      * Using mmap_mode
                - t_correct_index: np.ndarray of shape (num_validation_triplets,)
                      i-th row stores the index (0<= index < 1001) of the true tail entities
                      i.e., (h[i],r[i],t_candidate[i][t_correct_index[i]]) is the true triplet.

            - t,r->h: Given tail and relation, predict head entities
                - tr: np.ndarray of shape (num_validation_triplets, 2)
                      i-th row stores i-th (t,r)
                - h_candidate: np.ndarray of shape (num_validation_triplets, 1001)
                      i-th row stores i-th candidates for the head entities
                      * Using mmap_mode
                - h_correct_index: np.ndarray of shape (num_validation_triplets,)
                      i-th row stores the index (0<= index < 1001) of the true head entities
                      i.e., (h_candidate[i][h_correct_index[i]],r[i],t[i]) is the true triplet.
        '''
        if self._valid_dict is None:
            self._valid_dict = {}
            # h, r -> t
            self._valid_dict['h,r->t'] = {}
            self._valid_dict['h,r->t']['hr'] = np.load(osp.join(self.processed_dir, 'val_hr.npy'))
            self._valid_dict['h,r->t']['t_candidate'] = np.load(osp.join(self.processed_dir, 'val_t_candidate.npy'), mmap_mode='r')
            self._valid_dict['h,r->t']['t_correct_index'] = np.load(osp.join(self.processed_dir, 'val_t_correct_index.npy'))
            # t, r -> h
            self._valid_dict['t,r->h'] = {}
            self._valid_dict['t,r->h']['tr'] = np.load(osp.join(self.processed_dir, 'val_tr.npy'))
            self._valid_dict['t,r->h']['h_candidate'] = np.load(osp.join(self.processed_dir, 'val_h_candidate.npy'), mmap_mode='r')
            self._valid_dict['t,r->h']['h_correct_index'] = np.load(osp.join(self.processed_dir, 'val_h_correct_index.npy'))

        return self._valid_dict

    @property
    def test_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        '''
            Two test tasks: h,r->t and t,r->h

            - h,r->t: Given head and relation, predict target entities
                - hr: np.ndarray of shape (num_validation_triplets, 2)
                      i-th row stores i-th (h,r)
                - t_candidate: np.ndarray of shape (num_validation_triplets, 1001)
                      i-th row stores i-th candidates for the tail entities
                      * Using mmap_mode

            - t,r->h: Given tail and relation, predict head entities
                - tr: np.ndarray of shape (num_validation_triplets, 2)
                      i-th row stores i-th (t,r)
                - h_candidate: np.ndarray of shape (num_validation_triplets, 1001)
                      i-th row stores i-th candidates for the head entities
                      * Using mmap_mode

            * t_correct_index and h_correct_index are hidden
        '''
        if self._valid_dict is None:
            self._valid_dict = {}
            # h, r -> t
            self._valid_dict['h,r->t'] = {}
            self._valid_dict['h,r->t']['hr'] = np.load(osp.join(self.processed_dir, 'test_hr.npy'))
            self._valid_dict['h,r->t']['t_candidate'] = np.load(osp.join(self.processed_dir, 'test_t_candidate.npy'), mmap_mode='r')
            # t, r -> h
            self._valid_dict['t,r->h'] = {}
            self._valid_dict['t,r->h']['tr'] = np.load(osp.join(self.processed_dir, 'test_tr.npy'))
            self._valid_dict['t,r->h']['h_candidate'] = np.load(osp.join(self.processed_dir, 'test_h_candidate.npy'), mmap_mode='r')

        return self._valid_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class WikiKG90MEvaluator:
    def eval(self, input_dict):
        '''
            Format of input_dict storing the two tasks: hr->t, tr->h
            - 'hr->t'
                - t_pred_top10: np.ndarray of shape (num_eval_triplets, 10)
                    each element must < 1001
                    (i,j) represents the j-th prediction for i-th triplet
                    Only top10 prediction is taken into account
                - t_correct_index: np.ndarray of shape (num_eval_triplets,)

             - 'tr->h'
                - h_pred_top10: np.ndarray of shape (num_eval_triplets, 10)
                - h_correct_index: np.ndarray of shape (num_eval_triplets,)
        '''
        assert 'h,r->t' in input_dict and 't,r->h' in input_dict
        assert 't_pred_top10' in input_dict['h,r->t'] and 't_correct_index' in input_dict['h,r->t']
        assert 'h_pred_top10' in input_dict['t,r->h'] and 'h_correct_index' in input_dict['t,r->h']

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

        # t,r->h
        h_pred_top10 = input_dict['t,r->h']['h_pred_top10']
        h_correct_index = input_dict['t,r->h']['h_correct_index']
        if not isinstance(h_pred_top10, torch.Tensor):
            h_pred_top10 = torch.from_numpy(h_pred_top10)
        if not isinstance(h_correct_index, torch.Tensor):
            h_correct_index = torch.from_numpy(h_correct_index)
        assert h_pred_top10.shape[1] == 10 and len(h_pred_top10) == len(h_correct_index)
        assert (0 <= h_pred_top10).all() and (h_pred_top10 < 1001).all()
        assert (0 <= h_correct_index).all() and (h_correct_index < 1001).all()

        assert(len(h_pred_top10) == len(t_pred_top10))

        mrr_hr2t = self._calculate_mrr(t_correct_index.to(t_pred_top10.device), t_pred_top10)
        mrr_tr2h = self._calculate_mrr(h_correct_index.to(h_pred_top10.device), h_pred_top10)

        mrr = (mrr_hr2t + mrr_tr2h)/2

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
        assert 'h,r->t' in input_dict and 't,r->h' in input_dict
        assert 't_pred_top10' in input_dict['h,r->t']
        assert 'h_pred_top10' in input_dict['t,r->h']

        t_pred_top10 = input_dict['h,r->t']['t_pred_top10']
        h_pred_top10 = input_dict['t,r->h']['h_pred_top10']

        assert t_pred_top10.shape == (1359303, 10) and (0 <= t_pred_top10).all() and (t_pred_top10 < 1001).all()
        assert h_pred_top10.shape == (1359303, 10) and (0 <= h_pred_top10).all() and (h_pred_top10 < 1001).all()

        if isinstance(t_pred_top10, torch.Tensor):
            t_pred_top10 = t_pred_top10.cpu().numpy()
        t_pred_top10 = t_pred_top10.astype(np.int16)

        if isinstance(h_pred_top10, torch.Tensor):
            h_pred_top10 = h_pred_top10.cpu().numpy()
        h_pred_top10 = h_pred_top10.astype(np.int16)

        makedirs(dir_path)
        filename = osp.join(dir_path, 'ht_pred_wikikg90m')
        np.savez_compressed(filename, t_pred_top10=t_pred_top10, h_pred_top10=h_pred_top10)

if __name__ == '__main__':
    dataset = WikiKG90MDataset('/dfs/scratch1/weihuahu/ogb-lsc/datasets/wikikg90m')
    print(dataset)
    # print(dataset.num_entities)
    # print(dataset.entity_feat)
    # print(dataset.entity_feat.shape)
    # print(dataset.num_relations)
    # print(dataset.relation_feat)
    # print(dataset.relation_feat.shape)
    # print(dataset.train_hrt)
    # print(dataset.valid_dict)
    # print(dataset.test_dict)
    # print(dataset.valid_dict['t,r->h']['h_correct_index'].max())
    # print(dataset.valid_dict['t,r->h']['h_correct_index'].min())

    evaluator = WikiKG90MEvaluator()

    valid_dict = dataset.valid_dict
    t_correct_index = valid_dict['h,r->t']['t_correct_index']
    h_correct_index = valid_dict['t,r->h']['h_correct_index']

    h_pred_top10 = np.random.randint(0,1001, size=(len(h_correct_index), 10))
    t_pred_top10 = np.random.randint(0,1001, size=(len(t_correct_index), 10))

    input_dict = {}
    input_dict['h,r->t'] = {'t_correct_index': h_correct_index, 't_pred_top10': h_pred_top10}
    input_dict['t,r->h'] = {'h_correct_index': t_correct_index, 'h_pred_top10': t_pred_top10}
    result = evaluator.eval(input_dict)
    print(result)

    t_correct_index = torch.from_numpy(t_correct_index)
    h_correct_index = torch.from_numpy(h_correct_index)

    h_pred_top10 = torch.from_numpy(h_pred_top10)
    t_pred_top10 = torch.from_numpy(t_pred_top10)

    input_dict = {}
    input_dict['h,r->t'] = {'t_correct_index': h_correct_index, 't_pred_top10': h_pred_top10}
    input_dict['t,r->h'] = {'h_correct_index': t_correct_index, 'h_pred_top10': t_pred_top10}
    result = evaluator.eval(input_dict)
    print(result)

    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': np.random.randint(0,1001, size = (1359303, 10))}
    input_dict['t,r->h'] = {'h_pred_top10': np.random.randint(0,1001, size = (1359303, 10))}
    evaluator.save_test_submission(input_dict, 'result')
