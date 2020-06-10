### script for writing meta information of datasets into master.csv
### for graph property prediction datasets.
import pandas as pd

dataset_list = []
dataset_dict = {}

### start molecule dataset
dataset_dict["ogbg-molbace"] = {"num tasks": 1, "eval metric": "rocauc", "download_name": "bace"}
dataset_dict["ogbg-molbbbp"] = {"num tasks": 1, "eval metric": "rocauc", "download_name": "bbbp"}
dataset_dict["ogbg-molclintox"] = {"num tasks": 2, "eval metric": "rocauc", "download_name": "clintox"}
dataset_dict["ogbg-molmuv"] = {"num tasks": 17, "eval metric": "prcauc", "download_name": "muv"}
dataset_dict["ogbg-molpcba"] = {"num tasks": 128, "eval metric": "prcauc", "download_name": "pcba"}
dataset_dict["ogbg-molsider"] = {"num tasks": 27, "eval metric": "rocauc", "download_name": "sider"}
dataset_dict["ogbg-moltox21"] = {"num tasks": 12, "eval metric": "rocauc", "download_name": "tox21"}
dataset_dict["ogbg-moltoxcast"] = {"num tasks": 617, "eval metric": "rocauc", "download_name": "toxcast"}
dataset_dict["ogbg-molhiv"] = {"num tasks": 1, "eval metric": "rocauc", "download_name": "hiv"}
dataset_dict["ogbg-molesol"] = {"num tasks": 1, "eval metric": "rmse", "download_name": "esol"}
dataset_dict["ogbg-molfreesolv"] = {"num tasks": 1, "eval metric": "rmse", "download_name": "freesolv"}
dataset_dict["ogbg-mollipo"] = {"num tasks": 1, "eval metric": "rmse", "download_name": "lipophilicity"}
dataset_dict["ogbg-molchembl"] = {"num tasks": 1310, "eval metric": "rocauc", "download_name": "chembl"}
# dataset_dict["ogbg-molsars"] = {"num tasks": 1, "eval metric": "prcauc", "download_name": "sars"}
# dataset_dict["ogbg-molecoli"] = {"num tasks": 1, "eval metric": "rocauc", "download_name": "ecoli"}
# dataset_dict["ogbg-molsars2"] = {"num tasks": 1, "eval metric": "prcauc", "download_name": "pseudomonas"}

mol_dataset_list = list(dataset_dict.keys())

for nme in mol_dataset_list:
    download_folder_name = dataset_dict[nme]["download_name"]
    dataset_dict[nme]["version"] = 1
    dataset_dict[nme]["url"] = "https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/" + download_folder_name + ".zip"
    dataset_dict[nme]["add_inverse_edge"] = True
    dataset_dict[nme]["data type"] = "mol"
    dataset_dict[nme]["has_node_attr"] = True
    dataset_dict[nme]["has_edge_attr"] = True

    if dataset_dict[nme]["eval metric"] == "rmse":
        dataset_dict[nme]["task type"] = "regression"
        dataset_dict[nme]["num classes"] = -1 # num classes is not defined for regression datasets.
    else:
        dataset_dict[nme]["task type"] = "binary classification"
        dataset_dict[nme]["num classes"] = 2

    dataset_dict[nme]["split"] = "scaffold"

    dataset_dict[nme]["additional node files"] = 'None'
    dataset_dict[nme]['additional edge files'] = 'None'

    # if not nme == 'ogbg-molsars2':
    #     dataset_dict[nme]["split"] = "scaffold"
    # else:
    #     dataset_dict[nme]["split"] = "hidden"

dataset_list.extend(mol_dataset_list)

### end molecule dataset

### add ppi dataset (medium)
name = "ogbg-ppa"
dataset_dict[name] = {"eval metric": "acc"}
dataset_dict[name]["download_name"] = "ogbg_ppi_medium"
dataset_dict[name]["version"] = 1
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/graphproppred/" + dataset_dict[name]["download_name"] + ".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["split"] = "species"
dataset_dict[name]["num tasks"] = 1
dataset_dict[name]["has_node_attr"] = False
dataset_dict[name]["has_edge_attr"] = True
dataset_dict[name]["task type"] = "multiclass classification"
dataset_dict[name]["num classes"] = 37
dataset_dict[name]["additional node files"] = 'None'
dataset_dict[name]['additional edge files'] = 'None'


### add ppi dataset (medium)
name = "ogbg-code"
dataset_dict[name] = {"eval metric": "F1"}
dataset_dict[name]["download_name"] = "code"
dataset_dict[name]["version"] = 1
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/graphproppred/" + dataset_dict[name]["download_name"] + ".zip"
dataset_dict[name]["add_inverse_edge"] = False 
dataset_dict[name]["split"] = "project"
dataset_dict[name]["num tasks"] = 1
dataset_dict[name]["has_node_attr"] = True
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["task type"] = "sequence prediction"
dataset_dict[name]["num classes"] = -1
dataset_dict[name]["additional node files"] = 'node_is_attributed,node_dfs_order,node_depth'
dataset_dict[name]['additional edge files'] = 'None'


df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv("master.csv")