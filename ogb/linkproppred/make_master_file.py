### script for writing meta information of datasets into master.csv
### for link property prediction datasets.
import pandas as pd

dataset_dict = {}
dataset_list = []

### add meta-information about protein function prediction task
name = "ogbl-ppa"
dataset_dict[name] = {"task type": "link prediction"}
dataset_dict[name]["download_name"] = "ppassoc_v4"
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["has_node_attr"] = True
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "throughput"
dataset_dict[name]["num nodes"] = 576289

### add meta-information about amazon review rating prediction task
name = "ogbl-reviews-book"
dataset_dict[name] = {"task type": "link regression"}
dataset_dict[name]["download_name"] = "reviews_book_v2"
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["has_node_attr"] = True
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "time"
dataset_dict[name]["num nodes"] = 2560437

name = "ogbl-reviews-groc"
dataset_dict[name] = {"task type": "link regression"}
dataset_dict[name]["download_name"] = "reviews_grocery_v2"
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["has_node_attr"] = True
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "time"
dataset_dict[name]["num nodes"] = 168816

df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv("master.csv")
