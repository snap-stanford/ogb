# -*- coding: utf-8 -*-
#
# infer_emb.py
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

from .utils import load_entity_data, load_raw_emb_data, load_raw_emb_mapping
from .models.infer import EmbSimInfer

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--mfile', type=str, default=None,
                          help='ID mapping file.')
        self.add_argument('--emb_file', type=str, default=None,
                          help='Numpy file containing the embeddings.')
        self.add_argument('--format', type=str,
                          help='The format of input data'\
                                'l_r: two list of objects are provided as left objects and right objects.\n' \
                                'l_*: one list of objects is provided as left objects list and treat all objects in emb_file as right objects\n'
                                '*_r: one list of objects is provided as right objects list and treat all objects in emb_file as left objects\n'
                                '*: treat all objects in the emb_file as both left objects and right objects')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used to provide necessary files containing the requried data ' \
                               'according to the format, e.g., for l_r, two files are required as left_data and right_data; ' \
                               'for l_*, only one file is required; for *, no file is required')
        self.add_argument('--raw_data', default=False, action='store_true',
                          help='whether the data profiled in data_files is in the raw object naming space, e.g. string name of the entity' \
                                'or in DGL-KE converted integer id space \n' \
                                'If True, the data is in the original naming space and the inference program will do the id translation' \
                                'according to id mapping files generated during the training progress. \n' \
                                'If False, the data is just interger ids and it is assumed that user has already done the id translation')
        self.add_argument('--exec_mode', type=str, default='all',
                          help='How to calculate scores for element pairs and calculate topK: \n' \
                               'pairwise: both left and right objects are provided with the same length N, and we will calculate the similarity pair by pair:'\
                                    'result = topK([score(l_i, r_i)]) for i in N, the result shape will be (K,)'
                               'all: both left and right objects are provided as L and R, and we calculate all possible combinations of (l_i, r_j):' \
                                    'result = topK([[score(l_i, rj) for l_i in L] for r_j in R]), the result shape will be (K,)\n'
                               'batch_left: both left and right objects are provided as L and R,, and we calculate topK for each element in L:' \
                                    'result = topK([score(l_i, r_j) for r_j in R]) for l_j in L, the result shape will be (sizeof(L), K)\n')
        self.add_argument('--topK', type=int, default=10,
                          help='How many results are returned')
        self.add_argument('--sim_func', type=str, default='cosine',
                          help='What kind of similarity function is used in ranking and will be output: \n' \
                                'cosine: use cosine similarity, score = $\frac{x \cdot y}{||x||_2||y||_2}$' \
                                'l2: use l2 similarity, score = -$||x - y||_2$ \n' \
                                'l1: use l1 similarity, score = -$||x - y||_1$ \n' \
                                'dot: use dot product similarity, score = $x \cdot y$ \n' \
                                'ext_jaccard: use extended jaccard similarity, score = $\frac{x \cdot y}{||x||_{2}^{2} + ||y||_{2}^{2} - x \cdot y}$ \n')
        self.add_argument('--output', type=str, default='result.tsv',
                          help='Where to store the result, should be a single file')
        self.add_argument('--gpu', type=int, default=-1,
                          help='GPU device to use in inference, -1 means CPU')

def main():
    args = ArgParser().parse_args()
    assert args.emb_file != None, 'emb_file should be provided for entity embeddings'

    data_files = args.data_files
    if args.format == 'l_r':
        if args.raw_data:
            head, id2e_map, e2id_map = load_raw_emb_data(file=data_files[0],
                                                         map_f=args.mfile)
            tail, _, _ = load_raw_emb_data(file=data_files[1],
                                           e2id_map=e2id_map)
        else:
            head = load_entity_data(data_files[0])
            tail = load_entity_data(data_files[1])
    elif args.format == 'l_*':
        if args.raw_data:
            head, id2e_map, e2id_map = load_raw_emb_data(file=data_files[0],
                                                         map_f=args.mfile)
        else:
            head = load_entity_data(data_files[0])
        tail = load_entity_data()
    elif args.format == '*_r':
        if args.raw_data:
            tail, id2e_map, e2id_map = load_raw_emb_data(file=data_files[0],
                                                         map_f=args.mfile)
        else:
            tail = load_entity_data(data_files[0])
        head = load_entity_data()
    elif args.format == '*':
        if args.raw_data:
            id2e_map = load_raw_emb_mapping(map_f=args.mfile)
        head = load_entity_data()
        tail = load_entity_data()

    if args.exec_mode == 'pairwise':
        pairwise = True
        bcast = False
    elif args.exec_mode == 'batch_left':
        pairwise = False
        bcast = True
    elif args.exec_mode == 'all':
        pairwise = False
        bcast = False
    else:
        assert False, 'Unknow execution model'

    model = EmbSimInfer(args.gpu, args.emb_file, args.sim_func)
    model.load_emb()
    result = model.topK(head, tail, bcast=bcast, pair_ws=pairwise, k=args.topK)

    with open(args.output, 'w+') as f:
        f.write('left\tright\tscore\n')
        for res in result:
            hl, tl, sl = res
            hl = hl.tolist()
            tl = tl.tolist()
            sl = sl.tolist()

            for h, t, s in zip(hl, tl, sl):
                if args.raw_data:
                    h = id2e_map[h]
                    t = id2e_map[t]
                f.write('{}\t{}\t{}\n'.format(h, t, s))
    print('Inference Done')
    print('The result is saved in {}'.format(args.output))

if __name__ == '__main__':
    main()
