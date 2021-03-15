# -*- coding: utf-8 -*-
#
# ke_model.py
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
"""
Knowledge Graph Embedding Model
1. TransE_1
2. TransE_2
3. TransR
4. RESCAL
5. DistMult
6. ComplEx
7. RotatE
"""
import os
from abc import abstractmethod, ABCMeta
import numpy as np
import dgl
import torch as th

from .pytorch.tensor_models import logsigmoid
from .pytorch.tensor_models import none
from .pytorch.tensor_models import get_device
from .pytorch.tensor_models import norm
from .pytorch.tensor_models import get_scalar
from .pytorch.tensor_models import reshape
from .pytorch.tensor_models import cuda
from .pytorch.tensor_models import ExternalEmbedding
from .pytorch.tensor_models import InferEmbedding
from .pytorch.score_fun import *
from .pytorch.ke_tensor import KGEmbedding
from .pytorch.tensor_models import cosine_dist
from .pytorch.tensor_models import l2_dist
from .pytorch.tensor_models import l1_dist
from .pytorch.tensor_models import dot_dist
from .pytorch.tensor_models import extended_jaccard_dist
from .pytorch.tensor_models import floor_divide

EMB_INIT_EPS = 2.0
DEFAULT_INFER_BATCHSIZE = 1024

class BasicGEModel(object):
    """ Basic Graph Embeding Model
    """
    def __init__(self, device, model_name, score_func):
        self._g = None
        self._model_name = model_name
        self._device = device
        self._entity_emb = KGEmbedding(device)
        self._relation_emb = KGEmbedding(device)
        self._score_func = score_func

    def attach_graph(self, g, etid_field='tid', ntid_filed='ntid'):
        """ Attach dataset into Graph Embedding Model

        Parameter
        ----------
        g: DGLGraph
            Input data for knowledge graph
        etid_field: str
            Edge feature name storing the edge type id
        ntid_filed: str
            Node feature name storing the node type id

        Note
        ----
        If the input graph is DGLGraph, we assume that it uses a homogeneous graph
        to represent the heterogeneous graph. The edge type id is stored in etid_field
        and the node type id is stored in ntid_filed.
        """
        self._etid_field = etid_field
        self._ntid_filed = ntid_filed
        assert isinstance(g, dgl.DGLGraph)
        self._g = g

    def load(self, model_path):
        """ Load Graph Embedding Model from model_path.

        The default entity embeding file is entity.npy.
        The default relation embedding file is relation.npy.

        Parameter
        ---------
        model_path : str
            Path to store the model information
        """
        pass

    def save(self, model_path):
        """ Save Graph Embedding Model into model_path.

        All model related data are saved under model_path.
        The default entity embeding file is entity.npy.
        The default relation embedding file is relation.npy.

        Parameter
        ---------
        model_path : str
            Path to store the model information
        """
        assert False, 'Not support training now'

    def fit(self):
        """ Start training
        """
        assert False, 'Not support training now'

    def eval(self):
        """ Start evaluation
        """
        assert False, 'Not support evaluation now'

    def _infer_score_func(self, head, rel, tail, triplet_wise=False, batch_size=DEFAULT_INFER_BATCHSIZE):
        head_emb = self.entity_embed[head]
        rel_emb = self.relation_embed[rel]
        tail_emb = self.entity_embed[tail]

        num_head = head.shape[0]
        num_rel = rel.shape[0]
        num_tail = tail.shape[0]

        score = []
        if triplet_wise:
            # triplet wise score: head, relation and tail tensor have the same length N,
            # for i in range(N):
            #     result.append(score(head[i],rel[i],tail[i]))
            class FakeEdge(object):
                def __init__(self, head_emb, rel_emb, tail_emb, device=-1):
                    self._hobj = {}
                    self._robj = {}
                    self._tobj = {}
                    self._hobj['emb'] = head_emb.to(device)
                    self._robj['emb'] = rel_emb.to(device)
                    self._tobj['emb'] = tail_emb.to(device)

                @property
                def src(self):
                    return self._hobj

                @property
                def dst(self):
                    return self._tobj

                @property
                def data(self):
                    return self._robj

            # calculate scores in mini-batches
            # so we can use GPU to accelerate the speed with avoiding GPU OOM
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                sr_emb = rel_emb[i * batch_size : (i + 1) * batch_size \
                                                  if (i + 1) * batch_size < num_head \
                                                  else num_head]
                st_emb = tail_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                edata = FakeEdge(sh_emb, sr_emb, st_emb, self._device)
                score.append(self._score_func.edge_func(edata)['score'].to(th.device('cpu')))
            score = th.cat(score, dim=0)
            return score
        else:
            # head, relation and tail tensors has different size
            # for h_i in range(head):
            #     for r_j in range(relation):
            #         for t_k in range(tail):
            #             result.append(score(h_i, r_j, t_k))
            # The result will have shape (len(head), len(relation), len(tail))
            rel_emb = rel_emb.to(self._device)

            # calculating scores using mini-batch, the default batchsize if 1024
            # This can avoid OOM when using GPU
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                s_score = []
                sh_emb = sh_emb.to(self._device)
                for j in range((num_tail + batch_size - 1) // batch_size):
                    st_emb = tail_emb[j * batch_size : (j + 1) * batch_size \
                                                       if (j + 1) * batch_size < num_tail \
                                                       else num_tail]
                    st_emb = st_emb.to(self._device)
                    s_score.append(self._score_func.infer(sh_emb, rel_emb, st_emb).to(th.device('cpu')))
                score.append(th.cat(s_score, dim=2))
            score = th.cat(score, dim=0)
            return th.reshape(score, (num_head, num_rel, num_tail))

    def _exclude_pos(self, sidx, score, idx, head, rel, tail, topk, exec_mode, exclude_mode):
        g = self.graph
        num_triples = idx.shape[0]
        num_head = 1 if exec_mode == 'batch_head' else head.shape[0]
        num_rel = 1 if exec_mode == 'batch_rel' else rel.shape[0]
        num_tail = 1 if exec_mode == 'batch_tail' else tail.shape[0]

        res_head = []
        res_rel = []
        res_tail = []
        res_score = []
        result = []
        if exclude_mode == 'exclude':
            # exclude existing edges
            cur_k = 0
            batch_size = topk
            while (cur_k < num_triples):
                cur_sidx = sidx[cur_k:cur_k + batch_size if cur_k + batch_size < num_triples else num_triples]
                cur_score = score[cur_k:cur_k + batch_size if cur_k + batch_size < num_triples else num_triples]
                cur_idx = idx[cur_sidx]

                if exec_mode == 'triplet_wise':
                    cur_head = head[cur_idx]
                    cur_rel = rel[cur_idx]
                    cur_tail = tail[cur_idx]
                elif exec_mode == 'all':
                    tail_idx = cur_idx % num_tail
                    cur_idx = floor_divide(cur_idx, num_tail)
                    rel_idx = cur_idx % num_rel
                    cur_idx = floor_divide(cur_idx, num_rel)
                    head_idx = cur_idx % num_head

                    cur_head = head[head_idx]
                    cur_rel = rel[rel_idx]
                    cur_tail = tail[tail_idx]
                elif exec_mode == 'batch_head':
                    tail_idx = cur_idx % num_tail
                    cur_idx = floor_divide(cur_idx, num_tail)
                    rel_idx = cur_idx % num_rel

                    cur_head = th.full((cur_sidx.shape[0],), head, dtype=head.dtype)
                    cur_rel = rel[rel_idx]
                    cur_tail = tail[tail_idx]
                elif exec_mode == 'batch_rel':
                    tail_idx = cur_idx % num_tail
                    cur_idx = floor_divide(cur_idx, num_tail)
                    head_idx = cur_idx % num_head

                    cur_head = head[head_idx]
                    cur_rel = th.full((cur_sidx.shape[0],), rel, dtype=rel.dtype)
                    cur_tail = tail[tail_idx]
                elif exec_mode == 'batch_tail':
                    rel_idx = cur_idx % num_rel
                    cur_idx = floor_divide(cur_idx, num_rel)
                    head_idx = cur_idx % num_head

                    cur_head = head[head_idx]
                    cur_rel = rel[rel_idx]
                    cur_tail = th.full((cur_sidx.shape[0],), tail, dtype=tail.dtype)

                # Find exising edges
                # It is expacted that the existing edges are much less than triples
                # The idea is: 1) we get existing edges using g.edge_ids
                #              2) sort edges according to source node id (O(nlog(n)), n is number of edges)
                #              3) sort candidate triples according to cur_head (O(mlog(m)), m is number of cur_head nodes)
                #              4) go over all candidate triples and compare with existing edges,
                #                 as both edges and candidate triples are sorted. filtering edges out
                #                 will take only O(n+m)
                #              5) sort the score again it taks O(klog(k))
                uid, vid, eid = g.edge_ids(cur_head, cur_tail, return_uv=True)
                rid = g.edata[self._etid_field][eid]

                for i in range(cur_head.shape[0]):
                    h = cur_head[i]
                    r = cur_rel[i]
                    t = cur_tail[i]

                    h_where = uid == h
                    t_where = vid[h_where] == t
                    r_where = rid[h_where][t_where]
                    edge_exist = False
                    if r_where.shape[0] > 0:
                        for c_r in r_where:
                            if c_r == r:
                                edge_exist = True
                                break

                    if edge_exist is False:
                        res_head.append(h)
                        res_rel.append(r)
                        res_tail.append(t)
                        res_score.append(cur_score[i])

                if len(res_head) >= topk:
                    break

                cur_k += batch_size
                batch_size = topk - len(res_head) # check more edges
                batch_size = 16 if batch_size < 16 else batch_size # avoid tailing issue
            res_head = th.tensor(res_head)
            res_rel = th.tensor(res_rel)
            res_tail = th.tensor(res_tail)
            res_score = th.tensor(res_score)
            sidx = th.argsort(res_score, dim=0, descending=True)
            sidx = sidx[:topk] if topk < sidx.shape[0] else sidx
            result.append((res_head[sidx],
                           res_rel[sidx],
                           res_tail[sidx],
                           res_score[sidx],
                           None))
        else:
            # including the existing edges in the result
            topk = topk if topk < num_triples else num_triples
            sidx = sidx[:topk]
            idx = idx[sidx]

            if exec_mode == 'triplet_wise':
                head = head[idx]
                rel = rel[idx]
                tail = tail[idx]
            elif exec_mode == 'all':
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                rel_idx = idx % num_rel
                idx = floor_divide(idx, num_rel)
                head_idx = idx % num_head

                head = head[head_idx]
                rel = rel[rel_idx]
                tail = tail[tail_idx]
            elif exec_mode == 'batch_head':
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                rel_idx = idx % num_rel

                head = th.full((topk,), head, dtype=head.dtype)
                rel = rel[rel_idx]
                tail = tail[tail_idx]
            elif exec_mode == 'batch_rel':
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                head_idx = idx % num_head

                head = head[head_idx]
                rel = th.full((topk,), rel, dtype=rel.dtype)
                tail = tail[tail_idx]
            elif exec_mode == 'batch_tail':
                rel_idx = idx % num_rel
                idx = floor_divide(idx, num_rel)
                head_idx = idx % num_head

                head = head[head_idx]
                rel = rel[rel_idx]
                tail = th.full((topk,), tail, dtype=tail.dtype)

            if exclude_mode == 'mask':
                # Find exising edges
                # It is expacted that the existing edges are much less than triples
                # The idea is: 1) we get existing edges using g.edge_ids
                #              2) sort edges according to source node id (O(nlog(n)), n is number of edges)
                #              3) sort candidate triples according to cur_head (O(mlog(m)), m is number of cur_head nodes)
                #              4) go over all candidate triples and compare with existing edges and mask them,
                #                 as both edges and candidate triples are sorted. filtering edges out
                #                 will take only O(n+m)
                uid, vid, eid = g.edge_ids(head, tail, return_uv=True)
                rid = g.edata[self._etid_field][eid]
                mask = th.full((head.shape[0],), False, dtype=th.bool)

                if len(uid) > 0:
                    for i in range(head.shape[0]):
                        h = head[i]
                        r = rel[i]
                        t = tail[i]

                        h_where = uid == h
                        t_where = vid[h_where] == t
                        r_where = rid[h_where][t_where]
                        if r_where.shape[0] > 0:
                            for c_r in r_where:
                                if c_r == r:
                                    mask[i] = True
                                    break

                result.append((head, rel, tail, score, mask))
            else:
                result.append((head, rel, tail, score, None))

        return result

    def _topk_exclude_pos(self, score, idx, head, rel, tail, topk, exec_mode, exclude_mode):
        """ Generate topk most relevent triplets and corresponding scores.

            It takes following steps:

              1) find topk elements
              2) sort topk elements in descending order
              3) call _exclude_pos if figure out existing edges
        """
        if exclude_mode == 'exclude':
            if idx.shape[0] < topk * 4: # TODO(xiangsx): Find a better value of topk * n
                topk_score, topk_sidx = th.topk(score, k=idx.shape[0], dim=0)
                sidx = th.argsort(topk_score, dim=0, descending=True)
                sidx = topk_sidx[sidx]
                result = self._exclude_pos(sidx=sidx,
                                           score=topk_score,
                                           idx=idx,
                                           head=head,
                                           rel=rel,
                                           tail=tail,
                                           topk=topk,
                                           exec_mode=exec_mode,
                                           exclude_mode=exclude_mode)
            else:
                topk_score, topk_sidx = th.topk(score, k= topk * 4, dim=0)
                sidx = th.argsort(topk_score, dim=0, descending=True)
                sidx = topk_sidx[sidx]
                result = self._exclude_pos(sidx=sidx,
                                           score=topk_score,
                                           idx=idx,
                                           head=head,
                                           rel=rel,
                                           tail=tail,
                                           topk=topk,
                                           exec_mode=exec_mode,
                                           exclude_mode=exclude_mode)
                if len(result) < topk:
                    sidx = th.argsort(score, dim=0, descending=True)
                    result = self._exclude_pos(sidx=sidx,
                                               score=score[sidx],
                                               idx=idx,
                                               head=head,
                                               rel=rel,
                                               tail=tail,
                                               topk=topk,
                                               exec_mode=exec_mode,
                                               exclude_mode=exclude_mode)
        else:
            topk = idx.shape[0] if idx.shape[0] < topk else topk
            topk_score, topk_sidx = th.topk(score, k=topk, dim=0)
            sidx = th.argsort(topk_score, dim=0, descending=True)
            sidx = topk_sidx[sidx]
            result = self._exclude_pos(sidx=sidx,
                                       score=topk_score,
                                       idx=idx,
                                       head=head,
                                       rel=rel,
                                       tail=tail,
                                       topk=topk,
                                       exec_mode=exec_mode,
                                       exclude_mode=exclude_mode)
        return result

    def link_predict(self, head=None, rel=None, tail=None, exec_mode='all', sfunc='none', topk=10, exclude_mode=None, batch_size=DEFAULT_INFER_BATCHSIZE):
        """ Predicts missing entities or relations in a triplet.

        Given head_id, relation_id and tail_id, return topk most relevent triplet.

        Parameters
        ----------
        head: th.Tensor
            A tensor of head entity id.

        rel: th.Tensor
            A tensor of relation id.

        tail: th.Tensor
            A tensor of tail entity id.

        exec_mode: str
            How to calculate scores for triplets and calculate topK:

              * triplet_wise: head, relation and tail lists have the same length N,
                and we calculate the similarity triplet by triplet:
                ``result = topK([score(h_i, r_i, t_i) for i in N])``,
                the result shape will be (K,)

              * all: three lists of head, relation and tail ids are provided as H, R and T,
                and we calculate all possible combinations of all triplets (h_i, r_j, t_k):
                ``result = topK([[[score(h_i, r_j, t_k) for each h_i in H] for each r_j in R] for each t_k in T])``,
                the result shape will be (K,)

              * batch_head: three lists of head, relation and tail ids are provided as H, R and T
                and we calculate topK for each element in head:
                ``result = topK([[score(h_i, r_j, t_k) for each r_j in R] for each t_k in T]) for each h_i in H``
                the result shape will be (sizeof(H), K)

              * batch_rel: three lists of head, relation and tail ids are provided as H, R and T,
                and we calculate topK for each element in relation:
                ``result = topK([[score(h_i, r_j, t_k) for each h_i in H] for each t_k in T]) for each r_j in R``,
                the result shape will be (sizeof(R), K)

              * batch_tail: three lists of head, relation and tail ids are provided as H, R and T,
                and we calculate topK for each element in tail:
                ``result = topK([[score(h_i, r_j, t_k) for each h_i in H] for each r_j in R]) for each t_k in T``,
                the result shape will be (sizeof(T), K)

        sfunc: str
            What kind of score is used in ranking and will be output:

              * none: $score = x$
              * logsigmoid: $score = log(sigmoid(x))

        topk: int
            Return top k results

        exclude_mode: str
            Whether to exclude positive edges:

            * None: Do not exclude positive edges.

            * 'mask': Return topk edges and a mask indicating which one is positive edge.

            * 'exclude': Exclude positive edges, the returned k edges will be missing edges in the graph.

        Return
        ------
        A list of (head_idx, rel_idx, tail_idx, score)
        """
        if head is None:
            head = th.arange(0, self.num_entity)
        else:
            head = th.tensor(head)
        if rel is None:
            rel = th.arange(0, self.num_rel)
        else:
            rel = th.tensor(rel)
        if tail is None:
            tail = th.arange(0, self.num_entity)
        else:
            tail = th.tensor(tail)

        num_head = head.shape[0]
        num_rel = rel.shape[0]
        num_tail = tail.shape[0]

        if sfunc == 'none':
            sfunc = none
        else:
            sfunc = logsigmoid

        # if exclude_mode is not None, we need a graph to do the edge filtering
        assert (self._g is not None) or (exclude_mode is None), \
            'If exclude_mode is not None, please use load_graph() to initialize ' \
            'a graph for edge filtering.'
        if exec_mode == 'triplet_wise':
            assert num_head == num_rel, \
                'For triplet wise exection mode, head, relation and tail lists should have same length'
            assert num_head == num_tail, \
                'For triplet wise exection mode, head, relation and tail lists should have same length'

            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail, triplet_wise=True, batch_size=batch_size)
                score = sfunc(raw_score)
                idx = th.arange(0, num_head)

            result = self._topk_exclude_pos(score=score,
                                            idx=idx,
                                            head=head,
                                            rel=rel,
                                            tail=tail,
                                            topk=topk,
                                            exec_mode=exec_mode,
                                            exclude_mode=exclude_mode)
        elif exec_mode == 'all':
            result = []
            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail)
                raw_score = th.reshape(raw_score, (head.shape[0]*rel.shape[0]*tail.shape[0],))
                score = sfunc(raw_score)
            idx = th.arange(0, num_head * num_rel * num_tail)

            result = self._topk_exclude_pos(score=score,
                                            idx=idx,
                                            head=head,
                                            rel=rel,
                                            tail=tail,
                                            topk=topk,
                                            exec_mode=exec_mode,
                                            exclude_mode=exclude_mode)
        elif exec_mode == 'batch_head':
            result = []
            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail)
            for i in range(num_head):
                score = sfunc(th.reshape(raw_score[i,:,:], (rel.shape[0]*tail.shape[0],)))
                idx = th.arange(0, num_rel * num_tail)

                res = self._topk_exclude_pos(score=score,
                                             idx=idx,
                                             head=head[i],
                                             rel=rel,
                                             tail=tail,
                                             topk=topk,
                                             exec_mode=exec_mode,
                                             exclude_mode=exclude_mode)

                result.append(res[0])
        elif exec_mode == 'batch_rel':
            result = []
            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail)
            for i in range(num_rel):
                score = sfunc(th.reshape(raw_score[:,i,:], (head.shape[0]*tail.shape[0],)))
                idx = th.arange(0, num_head * num_tail)

                res = self._topk_exclude_pos(score=score,
                                             idx=idx,
                                             head=head,
                                             rel=rel[i],
                                             tail=tail,
                                             topk=topk,
                                             exec_mode=exec_mode,
                                             exclude_mode=exclude_mode)

                result.append(res[0])
        elif exec_mode == 'batch_tail':
            result = []
            with th.no_grad():
                raw_score = self._infer_score_func(head, rel, tail)
            for i in range(num_tail):
                score = sfunc(th.reshape(raw_score[:,:,i], (head.shape[0]*rel.shape[0],)))
                idx = th.arange(0, num_head * num_rel)

                res = self._topk_exclude_pos(score=score,
                                             idx=idx,
                                             head=head,
                                             rel=rel,
                                             tail=tail[i],
                                             topk=topk,
                                             exec_mode=exec_mode,
                                             exclude_mode=exclude_mode)

                result.append(res[0])
        else:
            assert False, 'unknow execution mode type {}'.format(exec_mode)

        return result

    def _embed_sim(self, head, tail, emb, sfunc='cosine', bcast=False, pair_ws=False, topk=10):
        batch_size=DEFAULT_INFER_BATCHSIZE
        if head is None:
            head = th.arange(0, emb.shape[0])
        else:
            head = th.tensor(head)
        if tail is None:
            tail = th.arange(0, emb.shape[0])
        else:
            tail = th.tensor(tail)
        head_emb = emb[head]
        tail_emb = emb[tail]

        if sfunc == 'cosine':
            sim_func = cosine_dist
        elif sfunc == 'l2':
            sim_func = l2_dist
        elif sfunc == 'l1':
            sim_func = l1_dist
        elif sfunc == 'dot':
            sim_func = dot_dist
        elif sfunc == 'ext_jaccard':
            sim_func = extended_jaccard_dist

        if pair_ws is True:
            result = []
            # chunked cal score
            score = []
            num_head = head.shape[0]
            num_tail = tail.shape[0]

            # calculating scores using mini-batch, the default batchsize if 1024
            # This can avoid OOM when using GPU
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                sh_emb = sh_emb.to(self._device)
                st_emb = tail_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                st_emb = st_emb.to(self._device)
                score.append(sim_func(sh_emb, st_emb, pw=True).to(th.device('cpu')))
            score = th.cat(score, dim=0)

            topk_score, topk_sidx = th.topk(score,
                                            k=topk if score.shape[0] > topk else score.shape[0],
                                            dim=0)
            sidx = th.argsort(topk_score, dim=0, descending=True)
            sidx = topk_sidx[sidx]
            score = score[sidx]
            result.append((head[sidx],
                           tail[sidx],
                           score))
        else:
            num_head = head.shape[0]
            num_tail = tail.shape[0]

            # calculating scores using mini-batch, the default batchsize if 1024
            # This can avoid OOM when using GPU
            score = []
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                            if (i + 1) * batch_size < num_head \
                                            else num_head]
                sh_emb = sh_emb.to(self._device)
                s_score = []
                for j in range((num_tail + batch_size - 1) // batch_size):
                    st_emb = tail_emb[j * batch_size : (j + 1) * batch_size \
                                                    if (j + 1) * batch_size < num_tail \
                                                    else num_tail]
                    st_emb = st_emb.to(self._device)
                    s_score.append(sim_func(sh_emb, st_emb).to(th.device('cpu')))
                score.append(th.cat(s_score, dim=1))
            score = th.cat(score, dim=0)

            if bcast is False:
                result = []
                idx = th.arange(0, num_head * num_tail)
                score = th.reshape(score, (num_head * num_tail, ))

                topk_score, topk_sidx = th.topk(score,
                                                k=topk if score.shape[0] > topk else score.shape[0],
                                                dim=0)
                sidx = th.argsort(topk_score, dim=0, descending=True)
                score = topk_score[sidx]
                sidx = topk_sidx[sidx]
                idx = idx[sidx]
                tail_idx = idx % num_tail
                idx = floor_divide(idx, num_tail)
                head_idx = idx % num_head

                result.append((head[head_idx],
                               tail[tail_idx],
                               score))

            else: # bcast at head
                result = []
                for i in range(num_head):
                    i_score = score[i]

                    topk_score, topk_sidx = th.topk(i_score,
                                                    k=topk if i_score.shape[0] > topk else i_score.shape[0],
                                                    dim=0)
                    sidx = th.argsort(topk_score, dim=0, descending=True)
                    i_score = topk_score[sidx]
                    idx = topk_sidx[sidx]

                    result.append((th.full((topk,), head[i], dtype=head[i].dtype),
                                  tail[idx],
                                  i_score))

        return result

    def embed_sim(self, left=None, right=None, embed_type='entity', sfunc='cosine', bcast=False, pair_ws=False, topk=10):
        """ Finds the most similar entity/relation embeddings for
        some pre-defined similarity functions given a set of
        entities or relations.

        Parameters
        ----------
        left: th.Tensor
            A tensor of left object id.

        right: th.Tensor
            A tensor of right object id.

        embed_type: str
            Whether it is using entity embedding or relation embedding.
            If `entity`, it is entity embedding.
            If 'relation', it is relation embedding.

        sfunc: str
            What kind of similarity function is used in ranking and will be output:

              * cosine: use cosine similarity, score = $\frac{x \cdot y}{||x||_2||y||_2}$'

              * l2: use l2 similarity, score = -$||x - y||_2$

              * l1: use l1 similarity, score = -$||x - y||_1$

              * dot: use dot product similarity, score = $x \cdot y$

              * ext_jaccard: use extended jaccard similarity, score = $\frac{x \cdot y}{||x||_{2}^{2} + ||y||_{2}^{2} - x \cdot y}$

        bcast: bool
            If True, both left and right objects are provided as L and R,, and we calculate topK for each element in L:

                * 'result = topK([score(l_i, r_j) for r_j in R]) for l_j in L, the result shape will be (sizeof(L), K)

            Default: False

        pair_ws: bool
            If True, both left and right objects are provided with the same length N, and we will calculate the similarity pair by pair:

              * result = topK([score(l_i, r_i)]) for i in N, the result shape will be (K,)

            Default: False

        topk: int
            Return top k results

        Note
        ----
        If both bcast and pair_ws is False, both left and right objects are provided as L and R,
        and we calculate all possible combinations of (l_i, r_j):
        ``result = topK([[score(l_i, rj) for l_i in L] for r_j in R])``,
        the result shape will be (K,)

        Return
        ------
        A list of (left_idx, right_idx, sim_score)
        """
        if embed_type == 'entity':
            emb = self.entity_embed
        elif embed_type == 'relation':
            emb = self.relation_embed
        else:
            assert False, 'emb should entity or relation'

        return self._embed_sim(head=left,
                               tail=right,
                               emb=emb,
                               sfunc=sfunc,
                               bcast=bcast,
                               pair_ws=pair_ws,
                               topk=topk)

    @property
    def model_name(self):
        return self._model_name

    @property
    def entity_embed(self):
        return self._entity_emb.emb

    @property
    def relation_embed(self):
        return self._relation_emb.emb

    @property
    def num_entity(self):
        return -1 if self.entity_embed is None else self.entity_embed.shape[0]

    @property
    def num_rel(self):
        return -1 if self.relation_embed is None else self.relation_embed.shape[0]

    @property
    def graph(self):
        return self._g

class KGEModel(BasicGEModel):
    """ Basic Knowledge Graph Embedding Model
    """
    def __init__(self, device, model_name, score_func):
        super(KGEModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        entity_emb_file = 'entity.npy'
        relation_emb_file = 'relation.npy'
        self._entity_emb.load(model_path, entity_emb_file)
        self._relation_emb.load(model_path, relation_emb_file)
        self._score_func.load(model_path, self.model_name)

class TransEModel(KGEModel):
    """ TransE Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransE'
        score_func = TransEScore(gamma, 'l2')
        self._gamma = gamma
        super(TransEModel, self).__init__(device, model_name, score_func)

class TransE_l2Model(KGEModel):
    """ TransE_l2 Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransE_l2'
        score_func = TransEScore(gamma, 'l2')
        self._gamma = gamma
        super(TransE_l2Model, self).__init__(device, model_name, score_func)

class TransE_l1Model(KGEModel):
    """ TransE_l1 Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransE_l1'
        score_func = TransEScore(gamma, 'l1')
        self._gamma = gamma
        super(TransE_l1Model, self).__init__(device, model_name, score_func)

class TransRModel(KGEModel):
    """ TransR Model
    """
    def __init__(self, device, gamma):
        model_name = 'TransR'
        # TransR score initialization is done at fit or load model
        projection_emb = KGEmbedding(device)
        score_func = TransRScore(gamma, projection_emb, -1, -1)
        self._gamma = gamma
        super(TransRModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        super(TransRModel, self).load(model_path)
        self._score_func.relation_dim = self._relation_emb.emb.shape[1]
        self._score_func.entity_dim = self._entity_emb.emb.shape[1]

class DistMultModel(KGEModel):
    """ DistMult Model
    """
    def __init__(self, device):
        model_name = 'DistMult'
        score_func = DistMultScore()
        super(DistMultModel, self).__init__(device, model_name, score_func)

class ComplExModel(KGEModel):
    """ ComplEx Model
    """
    def __init__(self, device):
        model_name = 'ComplEx'
        score_func = ComplExScore()
        super(ComplExModel, self).__init__(device, model_name, score_func)

class RESCALModel(KGEModel):
    """ RESCAL Model
    """
    def __init__(self, device):
        model_name = 'RESCAL'
        score_func = RESCALScore(-1, -1)
        super(RESCALModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        super(RESCALModel, self).load(model_path)
        self._score_func.entity_dim = self._entity_emb.emb.shape[1]
        self._score_func.relation_dim = self._relation_emb.emb.shape[1] // self._score_func.entity_dim

class RotatEModel(KGEModel):
    """ RotatE Model
    """
    def __init__(self, device, gamma):
        model_name = 'RotatE'
        self._gamma = gamma
        score_func = RotatEScore(gamma, 0)
        super(RotatEModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        super(RotatEModel, self).load(model_path)
        # retrive emb_init, which is used in scoring func
        entity_dim = self._entity_emb.emb.shape[1]
        hidden_dim = entity_dim // 2
        emb_init = (self._gamma + EMB_INIT_EPS) / hidden_dim
        self._score_func.emb_init = emb_init

class GNNModel(BasicGEModel):
    """ Basic GNN Model
    """
    def __init__(self, device, model_name, gamma=0):
        if model_name == 'TransE' or model_name == 'TransE_l2':
            score_func = TransEScore(gamma, 'l2')
        elif model_name == 'TransE_l1':
            score_func = TransEScore(gamma, 'l1')
        elif model_name == 'DistMult':
            score_func = DistMultScore()
        else:
            assert model_name in ['TransE', 'TransE_l2', 'TransE_l1', 'DistMult'], \
                "For general purpose Scoring function for GNN, we only support TransE_l1, TransE_l2" \
                "DistMult, but {} is given.".format(model_name)

        super(GNNModel, self).__init__(device, model_name, score_func)

    def load(self, model_path):
        entity_emb_file = 'entity.npy'
        relation_emb_file = 'relation.npy'
        self._entity_emb.load(model_path, entity_emb_file)
        self._relation_emb.load(model_path, relation_emb_file)
