### script for writing meta information of datasets into master.csv
### for graph property prediction datasets.
import pandas as pd

dataset_list = []
dataset_dict = {}

### start molecule dataset
dataset_dict["ogbg-molbace"] = {"num tasks": 1, "task type": "binary classification", "download_name": "bace"}
dataset_dict["ogbg-molbbbp"] = {"num tasks": 1, "task type": "binary classification", "download_name": "bbbp"}
dataset_dict["ogbg-molclintox"] = {"num tasks": 2, "task type": "binary classification", "download_name": "clintox"}
dataset_dict["ogbg-molmuv"] = {"num tasks": 17, "task type": "binary classification", "download_name": "muv"}
dataset_dict["ogbg-molpcba"] = {"num tasks": 128, "task type": "binary classification", "download_name": "pcba"}
dataset_dict["ogbg-molsider"] = {"num tasks": 27, "task type": "binary classification", "download_name": "sider"}
dataset_dict["ogbg-moltox21"] = {"num tasks": 12, "task type": "binary classification", "download_name": "tox21"}
dataset_dict["ogbg-moltoxcast"] = {"num tasks": 617, "task type": "binary classification", "download_name": "toxcast"}
dataset_dict["ogbg-molhiv"] = {"num tasks": 1, "task type": "binary classification", "download_name": "hiv"}
dataset_dict["ogbg-molesol"] = {"num tasks": 1, "task type": "regression", "download_name": "esol"}
dataset_dict["ogbg-molfreesolv"] = {"num tasks": 1, "task type": "regression", "download_name": "freesolv"}
dataset_dict["ogbg-mollipo"] = {"num tasks": 1, "task type": "regression", "download_name": "lipophilicity"}

mol_dataset_list = list(dataset_dict.keys())

for nme in mol_dataset_list:
    download_folder_name = dataset_dict[nme]["download_name"]
    dataset_dict[nme]["url"] = "https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/" + download_folder_name + ".zip"
    dataset_dict[nme]["add_inverse_edge"] = True
    dataset_dict[nme]["data type"] = "mol"
    dataset_dict[nme]["split"] = "scaffold"
    dataset_dict[nme]["has_node_attr"] = True
    dataset_dict[nme]["has_edge_attr"] = True

dataset_list.extend(mol_dataset_list)

### end molecule dataset

### add ppi dataset (medium)
name = "ogbg-ppi"
dataset_dict[name] = {"task type": "multiclass classification"}
dataset_dict[name]["download_name"] = "ogbg_ppi_medium"
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/graphproppred/ogbg_ppi_medium.zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["split"] = "species"
dataset_dict[name]["num tasks"] = 1
dataset_dict[name]["has_node_attr"] = False
dataset_dict[name]["has_edge_attr"] = True



df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv("master.csv")