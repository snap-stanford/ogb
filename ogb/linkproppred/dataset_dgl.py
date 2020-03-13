import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from dgl.data.utils import load_graphs, save_graphs, Subset
import dgl
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_dgl import read_csv_graph_dgl

class DglLinkPropPredDataset(object):
    """Adapted from https://docs.dgl.ai/en/latest/_modules/dgl/data/chem/csv_dataset.html#CSVDataset"""
    def __init__(self, name, root = "dataset"):
        self.name = name ## original name, e.g., ogbl-ppa
        self.dir_name = "_".join(name.split("-")) + "_dgl" ## replace hyphen with underline, e.g., ogbl_ppa_dgl

        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        self.meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), "master.csv"), index_col = 0)
        if not self.name in self.meta_info:
            print(self.name)
            error_mssg = "Invalid dataset name {}.\n".format(self.name)
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(self.meta_info.keys())
            raise ValueError(error_mssg)

        self.download_name = self.meta_info[self.name]["download_name"] ## name of downloaded file, e.g., ppassoc

        self.task_type = self.meta_info[self.name]["task type"]

        super(DglLinkPropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if osp.exists(pre_processed_file_path):
            self.graph, _ = load_graphs(pre_processed_file_path)

        else:
            ### check download
            if not osp.exists(osp.join(self.root, "raw", "edge.csv.gz")):
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

            raw_dir = osp.join(self.root, "raw")

            ### pre-process and save
            add_inverse_edge = self.meta_info[self.name]["add_inverse_edge"] == "True"
            graph = read_csv_graph_dgl(raw_dir, add_inverse_edge = add_inverse_edge)[0]

            print('Saving...')
            save_graphs(pre_processed_file_path, graph, {})

            self.graph, _ = load_graphs(pre_processed_file_path)

    def get_edge_split(self):
        split_type = self.meta_info[self.name]["split"]
        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header = None).values
        valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header = None).values
        test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header = None).values

        if self.task_type == "link prediction":
            target_type = torch.long
        else:
            target_type = torch.float

        return {"train_edge": torch.tensor(train_idx[:,:2], dtype = torch.long), "train_edge_label": torch.tensor(train_idx[:,2], dtype = target_type),
                    "valid_edge": torch.tensor(valid_idx[:,:2], dtype = torch.long), "valid_edge_label": torch.tensor(valid_idx[:,2], dtype = target_type), 
                        "test_edge": torch.tensor(test_idx[:,:2], dtype = torch.long), "test_edge_label": torch.tensor(test_idx[:,2], dtype = target_type)}

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph[0]

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

if __name__ == "__main__":
    dgl_dataset = DglLinkPropPredDataset(name = "ogbl-reviews")
    splitted_edge = dgl_dataset.get_edge_split()
    print(dgl_dataset[0])
    print(splitted_edge)
