from torch_geometric.data import InMemoryDataset
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_csv_graph_pyg, read_csv_heterograph_pyg
from ogb.utils.torch_util import replace_numpy_with_torchtensor


class PygLinkPropPredDataset(InMemoryDataset):
    def __init__(self, name, root = "dataset", transform=None, pre_transform=None):
        self.name = name ## original name, e.g., ogbl-ppa
        self.dir_name = "_".join(name.split("-")) + "_pyg" ## replace hyphen with underline, e.g., ogbl_ppa_pyg

        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        self.meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), "master.csv"), index_col = 0)
        if not self.name in self.meta_info:
            print(self.name)
            error_mssg = "Invalid dataset name {}.\n".format(self.name)
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(self.meta_info.keys())
            raise ValueError(error_mssg)

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user. 
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info[self.name]['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.root)

        self.download_name = self.meta_info[self.name]["download_name"] ## name of downloaded file, e.g., ppassoc

        self.task_type = self.meta_info[self.name]["task type"]
        self.eval_metric = self.meta_info[self.name]["eval metric"]
        self.is_hetero = self.meta_info[self.name]["is hetero"] == "True"

        super(PygLinkPropPredDataset, self).__init__(self.root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_edge_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info[self.name]["split"]
            
        path = osp.join(self.root, "split", split_type)

        train = replace_numpy_with_torchtensor(torch.load(osp.join(path, "train.pt")))
        valid = replace_numpy_with_torchtensor(torch.load(osp.join(path, "valid.pt")))
        test = replace_numpy_with_torchtensor(torch.load(osp.join(path, "test.pt")))

        return {"train": train, "valid": valid, "test": test}

    @property
    def raw_file_names(self):
        if self.is_hetero:
            return ['num-node-dict.csv.gz', 'triplet-type-list.csv.gz']
        else:
            file_names = ["edge"]
            if self.meta_info[self.name]["has_node_attr"] == "True":
                file_names.append("node-feat")
            if self.meta_info[self.name]["has_edge_attr"] == "True":
                file_names.append("edge-feat")
            return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join("geometric_data_processed.pt")

    def download(self):
        url =  self.meta_info[self.name]["url"]
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
            data = read_csv_heterograph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)[0]
        else:
            data = read_csv_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)[0]

        data = data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
        

if __name__ == "__main__":

    pyg_dataset = PygLinkPropPredDataset(name = "ogbl-biokg")
    split_edge = pyg_dataset.get_edge_split()
    print(pyg_dataset[0])
    exit(-1)

    pyg_dataset = PygLinkPropPredDataset(name = "ogbl-ddi")
    split_edge = pyg_dataset.get_edge_split()
    print(pyg_dataset[0])
    print(pyg_dataset[0].num_nodes)
    print(split_edge['train'])
    print(split_edge['test'])
    pyg_dataset = PygLinkPropPredDataset(name = "ogbl-wikikg")
    split_edge = pyg_dataset.get_edge_split()
    print(pyg_dataset[0])
    print(pyg_dataset[0].num_nodes)
    print(split_edge['train'])
    print(split_edge['test'])
    pyg_dataset = PygLinkPropPredDataset(name = "ogbl-citation")
    split_edge = pyg_dataset.get_edge_split()
    print(split_edge['train'])
    print(split_edge['test'])
    pyg_dataset = PygLinkPropPredDataset(name = "ogbl-ppa")
    split_edge = pyg_dataset.get_edge_split()
    print(split_edge['train'])
    print(split_edge['test'])
    pyg_dataset = PygLinkPropPredDataset(name = "ogbl-collab")
    split_edge = pyg_dataset.get_edge_split()
    print(split_edge['train'])
    print(split_edge['test'])