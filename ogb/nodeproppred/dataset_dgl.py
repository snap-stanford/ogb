import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from dgl.data.utils import load_graphs, save_graphs, Subset
import dgl
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_dgl import read_csv_graph_dgl, read_csv_heterograph_dgl
from ogb.io.read_graph_raw import read_node_label_hetero, read_nodesplitidx_split_hetero
import pickle

class DglNodePropPredDataset(object):
    def __init__(self, name, root = "dataset"):
        self.name = name ## original name, e.g., ogbn-proteins
        self.dir_name = "_".join(name.split("-")) + "_dgl" ## replace hyphen with underline, e.g., ogbn_proteins_dgl

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
        self.num_classes = int(self.meta_info[self.name]["num classes"])
        self.is_hetero = self.meta_info[self.name]["is hetero"] == "True"

        super(DglNodePropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if osp.exists(pre_processed_file_path):
    
            if not self.is_hetero:
                self.graph, label_dict = load_graphs(pre_processed_file_path)
                self.labels = label_dict['labels']
            else:
                with open(pre_processed_file_path, 'rb') as f:
                    self.graph, self.labels = pickle.load(f)

        else:
            ### check if the downloaded file exists
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
                graph = read_csv_heterograph_dgl(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)[0]
                
                label_dict = read_node_label_hetero(raw_dir)

                # convert into torch tensor
                if "classification" in self.task_type:
                    for nodetype in label_dict.keys():
                        # detect if there is any nan
                        node_label = label_dict[nodetype]
                        if np.isnan(node_label).any():
                            label_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)
                        else:
                            label_dict[nodetype] = torch.from_numpy(node_label).to(torch.long)
                else:
                    for nodetype in label_dict.keys():
                        node_label = label_dict[nodetype]
                        label_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)

                with open(pre_processed_file_path, 'wb') as f:
                    pickle.dump(([graph], label_dict), f)

                with open(pre_processed_file_path, 'rb') as f:
                    self.graph, self.labels = pickle.load(f)

            else:
                graph = read_csv_graph_dgl(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)[0]

                ### adding prediction target
                node_label = pd.read_csv(osp.join(raw_dir, 'node-label.csv.gz'), compression="gzip", header = None).values

                if "classification" in self.task_type:
                    # detect if there is any nan
                    if np.isnan(node_label).any():
                        node_label = torch.from_numpy(node_label).to(torch.float32)
                    else:
                        node_label = torch.from_numpy(node_label).to(torch.long)
                else:
                    node_label = torch.from_numpy(node_label).to(torch.float32)

                label_dict = {"labels": node_label}

                save_graphs(pre_processed_file_path, graph, label_dict)

                self.graph, label_dict = load_graphs(pre_processed_file_path)
                self.labels = label_dict['labels']

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

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph[idx], self.labels

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

if __name__ == "__main__":
    dgl_dataset = DglNodePropPredDataset(name = "ogbn-proteins")
    print(dgl_dataset.num_classes)
    split_index = dgl_dataset.get_idx_split()
    print(dgl_dataset[0])
    print(split_index)
