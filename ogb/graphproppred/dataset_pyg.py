from torch_geometric.data import InMemoryDataset
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_csv_graph_pyg


class PygGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root = "dataset", transform=None):
        self.name = name ## original name, e.g., ogbg-mol-tox21
        self.dir_name = "_".join(name.split("-")) + "_pyg" ## replace hyphen with underline, e.g., ogbg_mol_tox21_pyg

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

        super(PygGraphPropPredDataset, self).__init__(self.root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self):
        split_type = self.meta_info[self.name]["split"]
        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header = None).values.T[0]

        return {"train": torch.tensor(train_idx, dtype = torch.long), "valid": torch.tensor(valid_idx, dtype = torch.long), "test": torch.tensor(test_idx, dtype = torch.long)}

    @property
    def raw_file_names(self):
        file_names = ["edge"]
        if self.meta_info[self.name]["has_node_attr"] == "True":
            file_names.append("node-feat")
        if self.meta_info[self.name]["has_edge_attr"] == "True":
            file_names.append("edge-feat")
        return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        url = self.meta_info[self.name]["url"]
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print("Stop downloading.")
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        ### read pyg graph list
        add_inverse_edge = self.meta_info[self.name]["add_inverse_edge"] == "True"
        data_list = read_csv_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge)
        
        graph_label = pd.read_csv(osp.join(self.raw_dir, "graph-label.csv.gz"), compression="gzip", header = None).values

        ### add target labels
        for i, g in enumerate(data_list):
            g.y = torch.tensor(graph_label[i]).view(1,-1)

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
        

if __name__ == "__main__":
    pyg_dataset = PygGraphPropPredDataset(name = "ogbg-mol-tox21")
    splitted_index = pyg_dataset.get_idx_split()
    print(pyg_dataset)
    print(pyg_dataset[0])
    print(pyg_dataset[0].edge_index)
    print(pyg_dataset[splitted_index["train"]])
    print(pyg_dataset[splitted_index["valid"]])
    print(pyg_dataset[splitted_index["test"]])