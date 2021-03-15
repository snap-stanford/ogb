# -*- coding: utf-8 -*-
#
# train.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time
import argparse
import numpy as np

import dgl.backend as F

backend = os.environ.get('DGLBACKEND', 'pytorch')
from .general_models import InferModel
if backend.lower() == 'mxnet':
    from .mxnet.tensor_models import logsigmoid
    from .mxnet.tensor_models import none
    from .mxnet.tensor_models import get_dev
    from .mxnet.tensor_models import cosine_dist
    from .mxnet.tensor_models import l2_dist
    from .mxnet.tensor_models import l1_dist
    from .mxnet.tensor_models import dot_dist
    from .mxnet.tensor_models import extended_jaccard_dist
    from .mxnet.tensor_models import floor_divide
    DEFAULT_INFER_BATCHSIZE = 256
else:
    from .pytorch.tensor_models import logsigmoid
    from .pytorch.tensor_models import none
    from .pytorch.tensor_models import get_dev
    from .pytorch.tensor_models import cosine_dist
    from .pytorch.tensor_models import l2_dist
    from .pytorch.tensor_models import l1_dist
    from .pytorch.tensor_models import dot_dist
    from .pytorch.tensor_models import extended_jaccard_dist
    from .pytorch.tensor_models import floor_divide
    DEFAULT_INFER_BATCHSIZE = 1024

class ScoreInfer(object):
    """ Calculate score of triplet (h, r, t) based on pretained KG embeddings
        using specified score_function
    Parameters
    ---------
    device : int
        Device to run the inference, -1 for CPU
    config : dict
        Containing KG model information
    model_path : str
        path storing the model (pretrained embeddings)
    score_func : str
        What kind of score is used,
            none: score = $x$
            logsigmoid: score $log(sigmoid(x))
    """
    def __init__(self, device, config, model_path, sfunc='none'):
        assert sfunc in ['none', 'logsigmoid'], 'score function should be none or logsigmoid'

        self.device = 'cpu' if device < 0 else device
        self.config = config
        self.model_path = model_path
        self.sfunc = sfunc
        if sfunc == 'none':
            self.score_func = none
        else:
            self.score_func = logsigmoid

    def load_model(self):
        config = self.config
        model_path = self.model_path
        # for none score func, use 0.
        # for logsigmoid use original gamma to make the score closer to 0.
        gamma=config['gamma'] if self.sfunc == 'logsigmoid' else 0.0
        model = InferModel(device=self.device,
                           model_name=config['model'],
                           hidden_dim=config['emb_size'],
                           double_entity_emb=config['double_ent'],
                           double_relation_emb=config['double_rel'],
                           gamma=gamma)
        dataset = config['dataset']
        model.load_emb(model_path, dataset)
        self.model = model

    def topK(self, head=None, rel=None, tail=None, exec_mode='all', k=10):
        if head is None:
            head = F.arange(0, self.model.num_entity)
        else:
            head = F.tensor(head)
        if rel is None:
            rel = F.arange(0, self.model.num_rel)
        else:
            rel = F.tensor(rel)
        if tail is None:
            tail = F.arange(0, self.model.num_entity)
        else:
            tail = F.tensor(tail)

        num_head = F.shape(head)[0]
        num_rel = F.shape(rel)[0]
        num_tail = F.shape(tail)[0]

        if exec_mode == 'triplet_wise':
            result = []
            assert num_head == num_rel, \
                'For triplet wise exection mode, head, relation and tail lists should have same length'
            assert num_head == num_tail, \
                'For triplet wise exection mode, head, relation and tail lists should have same length'

            raw_score = self.model.score(head, rel, tail, triplet_wise=True)
            score = self.score_func(raw_score)
            idx = F.arange(0, num_head)

            sidx = F.argsort(score, dim=0, descending=True)
            sidx = sidx[:k]
            score = score[sidx]
            idx = idx[sidx]

            result.append((F.asnumpy(head[idx]),
                           F.asnumpy(rel[idx]),
                           F.asnumpy(tail[idx]),
                           F.asnumpy(score)))
        elif exec_mode == 'all':
            result = []
            raw_score = self.model.score(head, rel, tail)
            score = self.score_func(raw_score)
            idx = F.arange(0, num_head * num_rel * num_tail)

            sidx = F.argsort(score, dim=0, descending=True)
            sidx = sidx[:k]
            score = score[sidx]
            idx = idx[sidx]

            tail_idx = idx % num_tail
            idx = floor_divide(idx, num_tail)
            rel_idx = idx % num_rel
            idx = floor_divide(idx, num_rel)
            head_idx = idx % num_head

            result.append((F.asnumpy(head[head_idx]),
                           F.asnumpy(rel[rel_idx]),
                           F.asnumpy(tail[tail_idx]),
                           F.asnumpy(score)))
        elif exec_mode == 'batch_head':
            result = []
            for i in range(num_head):
                raw_score = self.model.score(F.unsqueeze(head[i], 0), rel, tail)
                score = self.score_func(raw_score)
                idx = F.arange(0, num_rel * num_tail)

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                rel_idx = idx % num_rel

                result.append((np.full((k,), F.asnumpy(head[i])),
                               F.asnumpy(rel[rel_idx]),
                               F.asnumpy(tail[tail_idx]),
                               F.asnumpy(score)))
        elif exec_mode == 'batch_rel':
            result = []
            for i in range(num_rel):
                raw_score = self.model.score(head, F.unsqueeze(rel[i], 0), tail)
                score = self.score_func(raw_score)
                idx = F.arange(0, num_head * num_tail)

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                head_idx = idx % num_head

                result.append((F.asnumpy(head[head_idx]),
                               np.full((k,), F.asnumpy(rel[i])),
                               F.asnumpy(tail[tail_idx]),
                               F.asnumpy(score)))
        elif exec_mode == 'batch_tail':
            result = []
            for i in range(num_tail):
                raw_score = self.model.score(head, rel, F.unsqueeze(tail[i], 0))
                score = self.score_func(raw_score)
                idx = F.arange(0, num_head * num_rel)

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                idx = idx[sidx]
                rel_idx = idx % num_rel
                idx = floor_divide(idx, num_rel)
                head_idx = idx % num_head
                result.append((F.asnumpy(head[head_idx]),
                               F.asnumpy(rel[rel_idx]),
                               np.full((k,), F.asnumpy(tail[i])),
                               F.asnumpy(score)))
        else:
            assert False, 'unknow execution mode type {}'.format(exec_mode)

        return result

class EmbSimInfer():
    """ Calculate simularity of entity/relation embeddings based on pretained KG embeddings
    Parameters
    ---------
    device : int
        Device to run the inference, -1 for CPU
    emb_file : dict
        Containing embedding information
    sfunc : str
        What kind of score is used,
            cosine: score = $\frac{x \cdot y}{||x||_2||y||_2}$
            l2: score = $-||x - y||_2$
            l1: score = $-||x - y||_1$
            dot: score = $x \cdot y$
            ext_jaccard: score = $\frac{x \cdot y}{||x||_{2}^{2} + ||y||_{2}^{2} - x \cdot y}$
    """
    def __init__(self, device, emb_file, sfunc='cosine', batch_size=DEFAULT_INFER_BATCHSIZE):
        self.device = get_dev(device)
        self.emb_file = emb_file
        self.sfunc = sfunc
        if sfunc == 'cosine':
            self.sim_func = cosine_dist
        elif sfunc == 'l2':
            self.sim_func = l2_dist
        elif sfunc == 'l1':
            self.sim_func = l1_dist
        elif sfunc == 'dot':
            self.sim_func = dot_dist
        elif sfunc == 'ext_jaccard':
            self.sim_func = extended_jaccard_dist
        self.batch_size = batch_size

    def load_emb(self):
        self.emb = F.tensor(np.load(self.emb_file))

    def topK(self, head=None, tail=None, bcast=False, pair_ws=False, k=10):
        if head is None:
            head = F.arange(0, self.emb.shape[0])
        else:
            head = F.tensor(head)
        if tail is None:
            tail = F.arange(0, self.emb.shape[0])
        else:
            tail = F.tensor(tail)

        head_emb = self.emb[head]
        tail_emb = self.emb[tail]
        if pair_ws is True:
            result = []
            batch_size = self.batch_size
            # chunked cal score
            score = []
            num_head = head.shape[0]
            num_tail = tail.shape[0]
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                sh_emb = F.copy_to(sh_emb, self.device)
                st_emb = tail_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                st_emb = F.copy_to(st_emb, self.device)
                score.append(F.copy_to(self.sim_func(sh_emb, st_emb, pw=True), F.cpu()))
            score = F.cat(score, dim=0)

            sidx = F.argsort(score, dim=0, descending=True)
            sidx = sidx[:k]
            score = score[sidx]
            result.append((F.asnumpy(head[sidx]),
                           F.asnumpy(tail[sidx]),
                           F.asnumpy(score)))
        else:
            num_head = head.shape[0]
            num_tail = tail.shape[0]
            batch_size = self.batch_size

            # chunked cal score
            score = []
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                            if (i + 1) * batch_size < num_head \
                                            else num_head]
                sh_emb = F.copy_to(sh_emb, self.device)
                s_score = []
                for j in range((num_tail + batch_size - 1) // batch_size):
                    st_emb = tail_emb[j * batch_size : (j + 1) * batch_size \
                                                    if (j + 1) * batch_size < num_tail \
                                                    else num_tail]
                    st_emb = F.copy_to(st_emb, self.device)
                    s_score.append(F.copy_to(self.sim_func(sh_emb, st_emb), F.cpu()))
                score.append(F.cat(s_score, dim=1))
            score = F.cat(score, dim=0)

            if bcast is False:
                result = []
                idx = F.arange(0, num_head * num_tail)
                score = F.reshape(score, (num_head * num_tail, ))

                sidx = F.argsort(score, dim=0, descending=True)
                sidx = sidx[:k]
                score = score[sidx]
                sidx = sidx
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                head_idx = idx % num_head

                result.append((F.asnumpy(head[head_idx]),
                           F.asnumpy(tail[tail_idx]),
                           F.asnumpy(score)))

            else: # bcast at head
                result = []
                for i in range(num_head):
                    i_score = score[i]

                    sidx = F.argsort(i_score, dim=0, descending=True)
                    idx = F.arange(0, num_tail)
                    i_idx = sidx[:k]
                    i_score = i_score[i_idx]
                    idx = idx[i_idx]

                    result.append((np.full((k,), F.asnumpy(head[i])),
                                  F.asnumpy(tail[idx]),
                                  F.asnumpy(i_score)))

        return result

