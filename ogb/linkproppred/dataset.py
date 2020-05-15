import pandas as pd
import shutil, os
import os.path as osp
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw
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

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user. 
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info[self.name]['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.root)

        self.task_type = self.meta_info[self.name]["task type"]
        self.eval_metric = self.meta_info[self.name]["eval metric"]
        self.is_hetero = self.meta_info[self.name]["is hetero"] == "True"

        super(LinkPropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'data_processed')

        if osp.exists(pre_processed_file_path):
            self.graph = torch.load(pre_processed_file_path, 'rb')

        else:
            ### check download
            has_necessary_file_simple = osp.exists(osp.join(self.root, "raw", "edge.csv.gz")) and (not self.is_hetero)
            has_necessary_file_hetero = osp.exists(osp.join(self.root, "raw", "triplet-type-list.csv.gz")) and self.is_hetero

            has_necessary_file = has_necessary_file_simple or has_necessary_file_hetero

            if not has_necessary_file:
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

            if self.meta_info[self.name]["additional node files"] == 'None':
                additional_node_files = []
            else:
                additional_node_files = self.meta_info[self.name]["additional node files"].split(',')

            if self.meta_info[self.name]["additional edge files"] == 'None':
                additional_edge_files = []
            else:
                additional_edge_files = self.meta_info[self.name]["additional edge files"].split(',')

            if self.is_hetero:
                self.graph = read_csv_heterograph_raw(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)[0] # only a single graph

            else:
                self.graph = read_csv_graph_raw(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)[0] # only a single graph

            print('Saving...')
            torch.save(self.graph, pre_processed_file_path)

    def get_edge_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info[self.name]["split"]
            
        path = osp.join(self.root, "split", split_type)

        train = torch.load(osp.join(path, "train.pt"))
        valid = torch.load(osp.join(path, "valid.pt"))
        test = torch.load(osp.join(path, "test.pt"))

        return {"train": train, "valid": valid, "test": test}

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

if __name__ == "__main__":
    dataset = LinkPropPredDataset(name = "ogbl-biokg")
    split_edge = dataset.get_edge_split()
    print(dataset[0])
    print(split_edge['train'].keys())
    print(split_edge['test'].keys())

    for key, item in split_edge['train'].items():
        print(key)
        if isinstance(item, torch.Tensor):
            print(item.shape)
        else:
            print(len(item))

    print("")

    for key, item in split_edge['valid'].items():
        print(key)
        if isinstance(item, torch.Tensor):
            print(item.shape)
        else:
            print(len(item))

    print("")

    for key, item in split_edge['test'].items():
        print(key)
        if isinstance(item, torch.Tensor):
            print(item.shape)
        else:
            print(len(item))
