# coding=utf-8

import torch
import numpy as np
import pickle
import sys

TORCH_PICKLE_MAGIC_NUMBER = 0x1950a86a20f9469cfc6c
TORCH_PICKLE_PROTOCOL_VERSION = 1001


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


def load_pt(pt_path):
    pickle_load_args = {}
    if sys.version_info >= (3, 0) and 'encoding' not in pickle_load_args.keys():
        pickle_load_args['encoding'] = 'utf-8'

    with open(pt_path, 'rb') as f:

        magic_number = pickle.load(f, **pickle_load_args)
        if magic_number != TORCH_PICKLE_MAGIC_NUMBER:
            raise RuntimeError("Invalid magic number; corrupt file?")
        protocol_version = pickle.load(f, **pickle_load_args)
        if protocol_version != TORCH_PICKLE_PROTOCOL_VERSION:
            raise RuntimeError("Invalid protocol version: %s" % protocol_version)

        _sys_info = pickle.load(f, **pickle_load_args)
        unpickler = pickle.Unpickler(f, **pickle_load_args)
        # unpickler.persistent_load = persistent_load
        result = unpickler.load()
    return result