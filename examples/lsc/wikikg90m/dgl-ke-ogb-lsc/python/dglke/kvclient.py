# -*- coding: utf-8 -*-
#
# kvclient.py
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
import argparse
import time
import logging

import socket
if os.name != 'nt':
    import fcntl
    import struct

import dgl
import dgl.backend as F

import torch.multiprocessing as mp
from .train_pytorch import load_model, dist_train_test
from .utils import get_compatible_batch_size, CommonArgParser

from .train import prepare_save_path
from .dataloader import TrainDataset, NewBidirectionalOneShotIterator
from .dataloader import get_partition_dataset

WAIT_TIME = 10

class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore')
        self.add_argument('--num_client', type=int, default=1,
                          help='Number of client on each machine.')

def get_long_tail_partition(n_relations, n_machine):
    """Relation types has a long tail distribution for many dataset.
       So we need to average shuffle the data before we partition it.
    """
    assert n_relations > 0, 'n_relations must be a positive number.'
    assert n_machine > 0, 'n_machine must be a positive number.'

    partition_book = [0] * n_relations

    part_id = 0
    for i in range(n_relations):
        partition_book[i] = part_id
        part_id += 1
        if part_id == n_machine:
          part_id = 0

    return partition_book 

def local_ip4_addr_list():
    """Return a set of IPv4 address
    """
    nic = set()

    for ix in socket.if_nameindex():
        name = ix[1]
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', name[:15].encode("UTF-8")))[20:24])
        nic.add(ip)

    return nic

def get_local_machine_id(server_namebook):
    """Get machine ID via server_namebook
    """
    assert len(server_namebook) > 0, 'server_namebook cannot be empty.'

    res = 0
    for ID, data in server_namebook.items():
        machine_id = data[0]
        ip = data[1]
        if ip in local_ip4_addr_list():
            res = machine_id
            break

    return res

def get_machine_count(ip_config):
    """Get total machine count
    """
    with open(ip_config) as f:
        machine_count = len(f.readlines())

    return machine_count

def start_client(args):
    """Start kvclient for training
    """
    init_time_start = time.time()
    time.sleep(WAIT_TIME) # wait for launch script

    # We cannot support gpu distributed training yet
    args.gpu = [-1] 
    args.mix_cpu_gpu = False
    args.async_update = False
    # We don't use relation partition in distributed training yet
    args.rel_part = False 
    args.strict_rel_part = False
    args.soft_rel_part = False
    # We don't support validation in distributed training
    args.valid = False
    total_machine = get_machine_count(args.ip_config)
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    machine_id = get_local_machine_id(server_namebook)

    dataset, entity_partition_book, local2global = get_partition_dataset(
        args.data_path,
        args.dataset,
        machine_id)

    n_entities = dataset.n_entities
    n_relations = dataset.n_relations

    print('Partition %d n_entities: %d' % (machine_id, n_entities))
    print("Partition %d n_relations: %d" % (machine_id, n_relations))

    entity_partition_book = F.tensor(entity_partition_book)
    relation_partition_book = get_long_tail_partition(dataset.n_relations, total_machine)
    relation_partition_book = F.tensor(relation_partition_book)
    local2global = F.tensor(local2global)

    relation_partition_book.share_memory_()
    entity_partition_book.share_memory_()
    local2global.share_memory_()

    train_data = TrainDataset(dataset, args, ranks=args.num_client)

    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = dataset.n_entities
    args.batch_size = get_compatible_batch_size(args.batch_size, args.neg_sample_size)
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)

    args.num_workers = 8 # fix num_workers to 8
    train_samplers = []
    for i in range(args.num_client):
        train_sampler_head = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='head',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False,
                                                       rank=i)
        train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='tail',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False,
                                                       rank=i)
        train_samplers.append(NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                              args.neg_sample_size, args.neg_sample_size,
                                                              True, n_entities))

    dataset = None

    model = load_model(args, n_entities, n_relations)
    model.share_memory()

    print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))

    rel_parts = train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None
    cross_rels = train_data.cross_rels if args.soft_rel_part else None

    procs = []
    for i in range(args.num_client):
        proc = mp.Process(target=dist_train_test, args=(args,
                                                        model,
                                                        train_samplers[i],
                                                        entity_partition_book,
                                                        relation_partition_book,
                                                        local2global,
                                                        i,
                                                        rel_parts,
                                                        cross_rels))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()

def main():
    args = ArgParser().parse_args()
    prepare_save_path(args)
    start_client(args)   

if __name__ == '__main__':
    main()
