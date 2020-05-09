import pandas as pd
import shutil, os
import numpy as np
import os.path as osp
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_raw import read_csv_graph_raw
import torch

class GraphPropPredDataset(object):
    def __init__(self, name, root = "dataset"):
        self.name = name ## original name, e.g., ogbg-mol-tox21
        self.dir_name = "_".join(name.split("-")) ## replace hyphen with underline, e.g., ogbg_mol_tox21

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
        self.eval_metric = self.meta_info[self.name]["eval metric"]
        self.task_type = self.meta_info[self.name]["task type"]
        self.num_classes = self.meta_info[self.name]["num classes"]

        super(GraphPropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        raw_dir = osp.join(self.root, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'data_processed')

        if os.path.exists(pre_processed_file_path):
            loaded_dict = torch.load(pre_processed_file_path, 'rb')
            self.graphs, self.labels = loaded_dict['graphs'], loaded_dict['labels']

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

            if self.meta_info[self.name]["additional node files"] == 'None':
                additional_node_files = []
            else:
                additional_node_files = self.meta_info[self.name]["additional node files"].split(',')

            if self.meta_info[self.name]["additional edge files"] == 'None':
                additional_edge_files = []
            else:
                additional_edge_files = self.meta_info[self.name]["additional edge files"].split(',')

            self.graphs = read_csv_graph_raw(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)


            if self.task_type == 'sequence prediction':
                labels_joined = pd.read_csv(osp.join(raw_dir, "graph-label.csv.gz"), compression="gzip", header = None).values
                # need to split each element into subtokens
                self.labels = [str(labels_joined[i][0]).split(' ') for i in range(len(labels_joined))]
            else:
                self.labels = pd.read_csv(osp.join(raw_dir, "graph-label.csv.gz"), compression="gzip", header = None).values

            print('Saving...')
            torch.save({'graphs': self.graphs, 'labels': self.labels}, pre_processed_file_path)


    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info[self.name]["split"]
            
        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header = None).values.T[0]

        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    def __getitem__(self, idx):
        """Get datapoint with index"""

        if isinstance(idx, (int, np.integer)):
            return self.graphs[idx], self.labels[idx]

        raise IndexError(
            'Only integer is valid index (got {}).'.format(type(idx).__name__))

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


if __name__ == "__main__":
    dataset = GraphPropPredDataset(name = "ogbg-code")
    target_list = np.array([len(label) for label in dataset.labels])
    print(np.sum(target_list == 1)/ float(len(target_list)))
    print(np.sum(target_list == 2)/ float(len(target_list)))
    print(np.sum(target_list == 3)/ float(len(target_list)))

    from collections import Counter
    print(Counter(target_list))

    print(dataset.num_classes)
    split_index = dataset.get_idx_split()
    # print(dataset)
    # print(dataset[2])
    # print(split_index["train"])
    # print(split_index["valid"])
    # print(split_index["test"])


