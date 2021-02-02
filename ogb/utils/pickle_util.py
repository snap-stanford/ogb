# coding=utf-8

import pickle
import sys

TORCH_PICKLE_MAGIC_NUMBER = 0x1950a86a20f9469cfc6c
TORCH_PICKLE_PROTOCOL_VERSION = 1001


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def load_torch_pt(pt_path):
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