import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from dgl.data.utils import load_graphs, save_graphs, Subset
import dgl
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_dgl import read_csv_graph_dgl

class DglGraphPropPredDataset(object):
    """Adapted from https://docs.dgl.ai/en/latest/_modules/dgl/data/chem/csv_dataset.html#CSVDataset"""
    def __init__(self, name, root = "dataset"):
        self.name = name ## original name, e.g., ogbg-mol-tox21
        self.dir_name = "_".join(name.split("-")) + "_dgl" ## replace hyphen with underline, e.g., ogbg_mol_tox21_dgl

        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        self.meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), "master.csv"), index_col = 0)
        if not self.name in self.meta_info:
            print(self.name)
            error_mssg = "Invalid dataset name {}.\n".format(self.name)
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(self.meta_info.keys())
            raise ValueError(error_mssg)

        self.download_name = self.meta_info[self.name]["download_name"] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info[self.name]["num tasks"])
        self.task_type = self.meta_info[self.name]["task type"]

        super(DglGraphPropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        raw_dir = osp.join(self.root, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if os.path.exists(pre_processed_file_path):
            self.graphs, label_dict = load_graphs(pre_processed_file_path)
            self.labels = label_dict['labels']

        else:
            ### download
            url = self.meta_info[self.name]["url"]
            if decide_download(url):
                path = download_url(url, self.original_root)
                extract_zip(path, self.original_root)
                os.unlink(path)
                # delete folder if there exists
                try:
                    shutil.rmtree(self.root)
                except:
                    pass
                shutil.move(osp.join(self.original_root, self.download_name), self.root)
            else:
                print("Stop download.")
                exit(-1)

            ### preprocess
            add_inverse_edge = self.meta_info[self.name]["add_inverse_edge"] == "True"
            graphs = read_csv_graph_dgl(raw_dir, add_inverse_edge = add_inverse_edge)
            labels = torch.tensor(pd.read_csv(osp.join(raw_dir, "graph-label.csv.gz"), compression="gzip", header = None).values)

            print('Saving...')
            save_graphs(pre_processed_file_path, graphs, labels={'labels': labels})

            ### load preprocessed files
            self.graphs, label_dict = load_graphs(pre_processed_file_path)
            self.labels = label_dict['labels']


    def get_idx_split(self):
        split_type = self.meta_info[self.name]["split"]
        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header = None).values.T[0]

        return {"train": torch.tensor(train_idx, dtype = torch.long), "valid": torch.tensor(valid_idx, dtype = torch.long), "test": torch.tensor(test_idx, dtype = torch.long)}

    def __getitem__(self, idx):
        """Get datapoint with index"""

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(labels)

if __name__ == "__main__":
    dgl_dataset = DglGraphPropPredDataset(name = "ogbg-mol-tox21")
    splitted_index = dgl_dataset.get_idx_split()
    print(dgl_dataset)
    print(dgl_dataset[0])
    print(dgl_dataset[splitted_index["train"]])
    print(dgl_dataset[splitted_index["valid"]])
    print(dgl_dataset[splitted_index["test"]])


