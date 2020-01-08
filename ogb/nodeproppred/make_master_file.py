### script for writing meta information of datasets into master.csv
### for node property prediction datasets.
import pandas as pd

dataset_dict = {}
dataset_list = []

### add meta-information about protein function prediction task
name = "ogbn-proteins"
dataset_dict[name] = {"num tasks": 112, "task type": "binary classification"}
dataset_dict[name]["download_name"] = "proteinfunc_v2"
dataset_dict[name]["url"] = "https://ogb.stanford.edu/data/nodeproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["has_node_attr"] = False
dataset_dict[name]["has_edge_attr"] = True
dataset_dict[name]["split"] = "species"
dataset_dict[name]["num nodes"] = 132534

df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv("master.csv")