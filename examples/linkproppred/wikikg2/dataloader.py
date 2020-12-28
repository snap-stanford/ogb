#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, count, true_head, true_tail):
        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = count
        self.true_head = true_head
        self.true_tail = true_tail
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        positive_sample = [head, relation, tail]

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        # negative_sample_list = []
        # negative_sample_size = 0

        # while negative_sample_size < self.negative_sample_size:
        #     negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
        #     if self.mode == 'head-batch':
        #         mask = np.in1d(
        #             negative_sample, 
        #             self.true_head[(relation, tail)], 
        #             assume_unique=True, 
        #             invert=True
        #         )
        #     elif self.mode == 'tail-batch':
        #         mask = np.in1d(
        #             negative_sample, 
        #             self.true_tail[(head, relation)], 
        #             assume_unique=True, 
        #             invert=True
        #         )
        #     else:
        #         raise ValueError('Training batch mode %s not supported' % self.mode)
        #     negative_sample = negative_sample[mask]
        #     negative_sample_list.append(negative_sample)
        #     negative_sample_size += negative_sample.size

        
        # negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        # negative_sample = torch.from_numpy(negative_sample)
        # negative_sample = torch.from_numpy(np.random.randint(self.nentity, size=self.negative_sample_size))
        negative_sample = torch.randint(0, self.nentity, (self.negative_sample_size,))
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
class TestDataset(Dataset):
    def __init__(self, triples, args, mode, random_sampling):
        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.mode = mode
        self.random_sampling = random_sampling
        if random_sampling:
            self.neg_size = args.neg_size_eval_train

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        positive_sample = torch.LongTensor((head, relation, tail))

        if self.mode == 'head-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([head]), torch.from_numpy(self.triples['head_neg'][idx])])
            else:
                negative_sample = torch.cat([torch.LongTensor([head]), torch.randint(0, self.nentity, size=(self.neg_size,))])
        elif self.mode == 'tail-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([tail]), torch.from_numpy(self.triples['tail_neg'][idx])])
            else:
                negative_sample = torch.cat([torch.LongTensor([tail]), torch.randint(0, self.nentity, size=(self.neg_size,))])

        return positive_sample, negative_sample, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]

        return positive_sample, negative_sample, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data