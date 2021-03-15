# -*- coding: utf-8 -*-
#
# tensor_models.py
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
KG Sparse embedding
"""
import os
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

from .score_fun import *
from .. import *

def logsigmoid(val):
    max_elem = nd.maximum(0., -val)
    z = nd.exp(-max_elem) + nd.exp(-val - max_elem)
    return -(max_elem + nd.log(z))

def abs(val):
    return nd.abs(val)

def masked_select(input, mask):
    assert False, 'masked select for MXNet is not implemented'

none = lambda x : x
get_dev = lambda gpu : mx.gpu(gpu) if gpu >= 0 else mx.cpu()
get_device = lambda args : mx.gpu(args.gpu[0]) if args.gpu[0] >= 0 else mx.cpu()

norm_l1 = lambda x: nd.sum(nd.abs(x))
norm = lambda x, p: nd.sum(nd.abs(x) ** p)

get_scalar = lambda x: x.detach().asscalar()

reshape = lambda arr, x, y: arr.reshape(x, y)

cuda = lambda arr, gpu: arr.as_in_context(mx.gpu(gpu))

def l2_dist(x, y, pw=False):
    if pw is False:
        x = x.expand_dims(axis=1)
        y = y.expand_dims(axis=0)

    return -nd.norm(x-y, ord=2, axis=-1)

def l1_dist(x, y, pw=False):
    if pw is False:
        x = x.expand_dims(axis=1)
        y = y.expand_dims(axis=0)

    return -nd.norm(x-y, ord=1, axis=-1)

def dot_dist(x, y, pw=False):
    if pw is False:
        x = x.expand_dims(axis=1)
        y = y.expand_dims(axis=0)

    return nd.sum(x * y, axis=-1)

def cosine_dist(x, y, pw=False):
    score = dot_dist(x, y, pw)

    x = nd.norm(x, ord=2, axis=-1)
    y = nd.norm(y, ord=2, axis=-1)
    if pw is False:
        x = x.expand_dims(axis=1)
        y = y.expand_dims(axis=0)

    return score / (x * y)

def extended_jaccard_dist(x, y, pw=False):
    score = dot_dist(x, y, pw)

    x = nd.norm(x, ord=2, axis=-1)**2
    y = nd.norm(y, ord=2, axis=-1)**2
    if pw is False:
        x = x.expand_dims(axis=1)
        y = y.expand_dims(axis=0)

    return score / (x + y - score)

def floor_divide(input, other):
    return input / other

class InferEmbedding:
    def __init__(self, device):
        self.device = device

    def load(self, path, name):
        """Load embeddings.

        Parameters
        ----------
        path : str
            Directory to load the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name+'.npy')
        self.emb = mx.nd.array(np.load(file_name))

    def load_emb(self, emb_array):
        """Load embeddings from numpy array.

        Parameters
        ----------
        emb_array : numpy.array  or torch.tensor
            Embedding array in numpy array or torch.tensor
        """
        if isinstance(emb_array, np.ndarray):
            self.emb = mx.nd.array(emb_array)
        else:
            self.emb = emb_array

    def __call__(self, idx):
        return self.emb[idx]

class ExternalEmbedding:
    """Sparse Embedding for Knowledge Graph
    It is used to store both entity embeddings and relation embeddings.

    Parameters
    ----------
    args :
        Global configs.
    num : int
        Number of embeddings.
    dim : int
        Embedding dimention size.
    ctx : mx.ctx
        Device context to store the embedding.
    """
    def __init__(self, args, num, dim, ctx):
        self.gpu = args.gpu
        self.args = args
        self.trace = []

        self.emb = nd.empty((num, dim), dtype=np.float32, ctx=ctx)
        self.state_sum = nd.zeros((self.emb.shape[0]), dtype=np.float32, ctx=ctx)
        self.state_step = 0

    def init(self, emb_init):
        """Initializing the embeddings.

        Parameters
        ----------
        emb_init : float
            The intial embedding range should be [-emb_init, emb_init].
        """
        nd.random.uniform(-emb_init, emb_init,
                          shape=self.emb.shape, dtype=self.emb.dtype,
                          ctx=self.emb.context, out=self.emb)

    def share_memory(self):
        # TODO(zhengda) fix this later
        pass

    def __call__(self, idx, gpu_id=-1, trace=True):
        """ Return sliced tensor.

        Parameters
        ----------
        idx : th.tensor
            Slicing index
        gpu_id : int
            Which gpu to put sliced data in.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: True
        """
        if self.emb.context != idx.context:
            idx = idx.as_in_context(self.emb.context)
        data = nd.take(self.emb, idx)
        if gpu_id >= 0:
            data = data.as_in_context(mx.gpu(gpu_id))
        data.attach_grad()
        if trace:
            self.trace.append((idx, data))
        return data

    def update(self, gpu_id=-1):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        self.state_step += 1
        for idx, data in self.trace:
            grad = data.grad

            clr = self.args.lr
            #clr = self.args.lr / (1 + (self.state_step - 1) * group['lr_decay'])

            # the update is non-linear so indices must be unique
            grad_indices = idx
            grad_values = grad

            grad_sum = (grad_values * grad_values).mean(1)
            ctx = self.state_sum.context
            if ctx != grad_indices.context:
                grad_indices = grad_indices.as_in_context(ctx)
            if ctx != grad_sum.context:
                grad_sum = grad_sum.as_in_context(ctx)
            self.state_sum[grad_indices] += grad_sum
            std = self.state_sum[grad_indices]  # _sparse_mask
            if gpu_id >= 0:
                std = std.as_in_context(mx.gpu(gpu_id))
            std_values = nd.expand_dims(nd.sqrt(std) + 1e-10, 1)
            tmp = (-clr * grad_values / std_values)
            if tmp.context != ctx:
                tmp = tmp.as_in_context(ctx)
            # TODO(zhengda) the overhead is here.
            self.emb[grad_indices] = mx.nd.take(self.emb, grad_indices) + tmp
        self.trace = []

    def curr_emb(self):
        """Return embeddings in trace.
        """
        data = [data for _, data in self.trace]
        return nd.concat(*data, dim=0)

    def save(self, path, name):
        """Save embeddings.

        Parameters
        ----------
        path : str
            Directory to save the embedding.
        name : str
            Embedding name.
        """
        emb_fname = os.path.join(path, name+'.npy')
        np.save(emb_fname, self.emb.asnumpy())

    def load(self, path, name):
        """Load embeddings.

        Parameters
        ----------
        path : str
            Directory to load the embedding.
        name : str
            Embedding name.
        """
        emb_fname = os.path.join(path, name+'.npy')
        self.emb = nd.array(np.load(emb_fname))
