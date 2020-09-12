import torch
import numpy as np

def replace_numpy_with_torchtensor(obj):
    # assume obj comprises either list or dictionary
    # replace all the numpy instance with torch tensor.

    if isinstance(obj, dict):
        for key in obj.keys():
            if isinstance(obj[key], np.ndarray):
                obj[key] = torch.from_numpy(obj[key])
            else:
                replace_numpy_with_torchtensor(obj[key])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            if isinstance(obj[i], np.ndarray):
                obj[i] = torch.from_numpy(obj[i])
            else:
                replace_numpy_with_torchtensor(obj[i])

    # if the original input obj is numpy array
    elif isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    return obj


def all_numpy(obj):
    # Ensure everything is in numpy or int or float (no torch tensor)

    if isinstance(obj, dict):
        for key in obj.keys():
            all_numpy(obj[key])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            all_numpy(obj[i])
    else:
        if not isinstance(obj, (np.ndarray, int, float)):
            return False

    return True