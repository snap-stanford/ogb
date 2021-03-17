import os
import pickle
import json
import numpy as np
import sys
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
import pdb
from collections import defaultdict
import torch.nn.functional as F
import torch

# python save_test_submission.py $SAVE_PATH $NUM_PROC
if __name__ == '__main__':
    path = sys.argv[1]
    num_proc = int(sys.argv[2])

    all_file_names = os.listdir(path)
    test_file_names = [name for name in all_file_names if '.pkl' in name and 'test' in name]
    valid_file_names = [name for name in all_file_names if '.pkl' in name and 'valid' in name]
    steps = [int(name.split('.')[0].split('_')[-1]) for name in valid_file_names if 'valid_0' in name]
    steps.sort()
    evaluator = WikiKG90MEvaluator()
    device = torch.device('cpu')

    all_test_dicts = []
    best_valid_mrr = -1
    best_valid_idx = -1

    for i, step in enumerate(steps):
        valid_result_dict = defaultdict(lambda: defaultdict(list))
        test_result_dict = defaultdict(lambda: defaultdict(list))
        for proc in range(num_proc):
            valid_result_dict_proc = torch.load(os.path.join(path, "valid_{}_{}.pkl".format(proc, step)), map_location=device)
            test_result_dict_proc = torch.load(os.path.join(path, "test_{}_{}.pkl".format(proc, step)), map_location=device)
            for result_dict_proc, result_dict in zip([valid_result_dict_proc, test_result_dict_proc], [valid_result_dict, test_result_dict]):
                for key in result_dict_proc['h,r->t']:
                    result_dict['h,r->t'][key].append(result_dict_proc['h,r->t'][key].numpy())
        for result_dict in [valid_result_dict, test_result_dict]:
            for key in result_dict['h,r->t']:
                result_dict['h,r->t'][key] = np.concatenate(result_dict['h,r->t'][key], 0)
                    
        all_test_dicts.append(test_result_dict)
        metrics = evaluator.eval(valid_result_dict)
        metric = 'mrr'
        print("valid-{} at step {}: {}".format(metric, step, metrics[metric]))
        if metrics[metric] > best_valid_mrr:
            best_valid_mrr = metrics[metric]
            best_valid_idx = i

    best_test_dict = all_test_dicts[best_valid_idx]
    evaluator.save_test_submission(best_test_dict, path)



