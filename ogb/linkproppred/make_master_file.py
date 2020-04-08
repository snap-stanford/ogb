### script for writing meta information of datasets into master.csv
### for link property prediction datasets.
import pandas as pd

dataset_dict = {}
dataset_list = []

### add meta-information about protein function prediction task
name = "ogbl-ppa"
dataset_dict[name] = {"eval metric": "hits@100", "task type": "link prediction"}
dataset_dict[name]["download_name"] = "ppassoc_v5"
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["has_node_attr"] = True
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "throughput"
dataset_dict[name]["num nodes"] = 576289
dataset_dict[name]["additional node files"] = 'None'
dataset_dict[name]['additional edge files'] = 'None'

### add meta-information about protein function prediction task
name = "ogbl-collab"
dataset_dict[name] = {"eval metric": "hits@K", "task type": "link prediction"}
dataset_dict[name]["download_name"] = "collab"
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["has_node_attr"] = True
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "time"
dataset_dict[name]["num nodes"] = 235868
dataset_dict[name]["additional node files"] = 'None'
dataset_dict[name]['additional edge files'] = 'edge_weight,edge_year'

df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv("master.csv")
