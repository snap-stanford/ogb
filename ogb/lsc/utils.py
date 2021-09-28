import torch
import numpy as np

# Assigning the whole test into test-dev and test-challenge
def split_test(split_dict):
    if 'test' not in split_dict:
        assert 'test-whole' in split_dict
        assert 'test-dev' in split_dict
        assert 'test-challenge' in split_dict
        return 

    if isinstance(split_dict['test'], torch.Tensor):
        idx = torch.arange(len(split_dict['test']))
        dev_idx = torch.nonzero(idx % 5 < 3, as_tuple=True)[0]
        challenge_idx = torch.nonzero(~(idx % 5 < 3), as_tuple=True)[0]
    else:
        idx = np.arange(len(split_dict['test']))
        dev_idx = np.nonzero(idx % 5 < 3)[0]
        challenge_idx = np.nonzero(~(idx % 5 < 3))[0]

    split_dict['test-whole'] = split_dict['test']
    split_dict['test-dev'] = split_dict['test-whole'][dev_idx]
    split_dict['test-challenge'] = split_dict['test-whole'][challenge_idx]

    assert len(split_dict['test-dev']) + len(split_dict['test-whole'][challenge_idx]) == len(split_dict['test'])

    del split_dict['test']