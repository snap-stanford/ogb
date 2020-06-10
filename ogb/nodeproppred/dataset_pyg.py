from torch_geometric.data import InMemoryDataset
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_csv_graph_pyg, read_csv_heterograph_pyg
from ogb.io.read_graph_raw import read_node_label_hetero, read_nodesplitidx_split_hetero

class PygNodePropPredDataset(InMemoryDataset):
    def __init__(self, name, root = "dataset", transform=None, pre_transform=None):
        self.name = name ## original name, e.g., ogbn-proteins
        self.dir_name = "_".join(name.split("-")) + "_pyg" ## replace hyphen with underline, e.g., ogbn_proteins_pyg

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


        self.download_name = self.meta_info[self.name]["download_name"] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info[self.name]["num tasks"])
        self.task_type = self.meta_info[self.name]["task type"]
        self.eval_metric = self.meta_info[self.name]["eval metric"]
        self.__num_classes__ = int(self.meta_info[self.name]["num classes"])
        self.is_hetero = self.meta_info[self.name]["is hetero"] == "True"

        super(PygNodePropPredDataset, self).__init__(self.root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info[self.name]["split"]

        path = osp.join(self.root, "split", split_type)

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = torch.from_numpy(train_idx_dict[nodetype]).to(torch.long)
                valid_idx_dict[nodetype] = torch.from_numpy(valid_idx_dict[nodetype]).to(torch.long)
                test_idx_dict[nodetype] = torch.from_numpy(test_idx_dict[nodetype]).to(torch.long)

                return {"train": train_idx_dict, "valid": valid_idx_dict, "test": test_idx_dict}

        else:
            train_idx = torch.from_numpy(pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header = None).values.T[0]).to(torch.long)
            valid_idx = torch.from_numpy(pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header = None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header = None).values.T[0]).to(torch.long)

            return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    @property
    def num_classes(self):
        return self.__num_classes__

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

            node_label_dict = read_node_label_hetero(self.raw_dir)

            data.y_dict = {}
            if "classification" in self.task_type:
                for nodetype, node_label in node_label_dict.items():
                    # detect if there is any nan
                    if np.isnan(node_label).any():
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)
                    else:
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.long)
            else:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)

        else:
            data = read_csv_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)[0]

            ### adding prediction target
            node_label = pd.read_csv(osp.join(self.raw_dir, 'node-label.csv.gz'), compression="gzip", header = None).values

            if "classification" in self.task_type:
                # detect if there is any nan
                if np.isnan(node_label).any():
                    data.y = torch.from_numpy(node_label).to(torch.float32)
                else:
                    data.y = torch.from_numpy(node_label).to(torch.long)

            else:
                data.y = torch.from_numpy(node_label).to(torch.float32)

        data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
        

if __name__ == "__main__":
    pyg_dataset = PygNodePropPredDataset(name = "ogbn-mag")
    print(pyg_dataset[0])
    split_index = pyg_dataset.get_idx_split()
    # print(split_index)
    