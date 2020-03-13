import pandas as pd
import shutil, os
import os.path as osp
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_raw import read_csv_graph_raw
import torch
import numpy as np

class LinkPropPredDataset(object):
    def __init__(self, name, root = "dataset"):
        self.name = name ## original name, e.g., ogbl-ppa
        self.dir_name = "_".join(name.split("-")) ## replace hyphen with underline, e.g., ogbl_ppa

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

        super(LinkPropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'data_processed')

        if osp.exists(pre_processed_file_path):
            self.graph = torch.load(pre_processed_file_path, 'rb')

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
            self.graph = read_csv_graph_raw(raw_dir, add_inverse_edge = add_inverse_edge)[0] # only a single graph

            print('Saving...')
            torch.save(self.graph, pre_processed_file_path)

    def get_edge_split(self):
        split_type = self.meta_info[self.name]["split"]
        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header = None).values
        valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header = None).values
        test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header = None).values

        if self.task_type == "link prediction":
            target_type = np.int64
        else:
            target_type = np.float32

        return {"train_edge": train_idx[:,:2].astype(np.int64), "train_edge_label": train_idx[:,2].astype(target_type),
                    "valid_edge": valid_idx[:,:2].astype(np.int64), "valid_edge_label": valid_idx[:,2].astype(target_type), 
                        "test_edge": test_idx[:,:2].astype(np.int64), "test_edge_label": test_idx[:,2].astype(target_type)}

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

if __name__ == "__main__":
    dataset = LinkPropPredDataset(name = "ogbl-reviews-groc")
    splitted_edge = dataset.get_edge_split()
    print(dataset[0])
    print(splitted_edge)
