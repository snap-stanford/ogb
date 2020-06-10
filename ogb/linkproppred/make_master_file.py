### script for writing meta information of datasets into master.csv
### for link property prediction datasets.
import pandas as pd

dataset_dict = {}
dataset_list = []

### add meta-information about protein function prediction task
name = "ogbl-ppa"
dataset_dict[name] = {"eval metric": "hits@100", "task type": "link prediction"}
dataset_dict[name]["download_name"] = "ppassoc"
dataset_dict[name]["version"] = 1
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["has_node_attr"] = True
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "throughput"
dataset_dict[name]["additional node files"] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False

### add meta-information about author collaboration prediction task
name = "ogbl-collab"
dataset_dict[name] = {"eval metric": "hits@10", "task type": "link prediction"}
dataset_dict[name]["download_name"] = "collab"
dataset_dict[name]["version"] = 1
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["has_node_attr"] = True
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "time"
dataset_dict[name]["additional node files"] = 'None'
dataset_dict[name]['additional edge files'] = 'edge_weight,edge_year'
dataset_dict[name]['is hetero'] = False

### add meta-information about paper citation recommendation task
name = "ogbl-citation"
dataset_dict[name] = {"eval metric": "mrr", "task type": "link prediction"}
dataset_dict[name]["download_name"] = "citation"
dataset_dict[name]["version"] = 1
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = False 
dataset_dict[name]["has_node_attr"] = True
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "time"
dataset_dict[name]["additional node files"] = 'node_year'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False

### add meta-information about wikidata knowledge graph completion task
name = "ogbl-wikikg"
dataset_dict[name] = {"eval metric": "mrr", "task type": "KG completion"}
dataset_dict[name]["download_name"] = "wikikg"
dataset_dict[name]["version"] = 1
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = False 
dataset_dict[name]["has_node_attr"] = False
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "time"
dataset_dict[name]["additional node files"] = 'None'
dataset_dict[name]['additional edge files'] = 'edge_reltype'
dataset_dict[name]['is hetero'] = False

### add meta-information about drug drug interaction prediction task
name = "ogbl-ddi"
dataset_dict[name] = {"eval metric": "hits@10", "task type": "link prediction"}
dataset_dict[name]["download_name"] = "ddi"
dataset_dict[name]["version"] = 1
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = True 
dataset_dict[name]["has_node_attr"] = False
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "target"
dataset_dict[name]["additional node files"] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False

### add meta-information about biological knowledge graph completion task
name = "ogbl-biokg"
dataset_dict[name] = {"eval metric": "mrr", "task type": "KG completion"}
dataset_dict[name]["download_name"] = "biokg"
dataset_dict[name]["version"] = 1
dataset_dict[name]["url"] = "https://snap.stanford.edu/ogb/data/linkproppred/"+dataset_dict[name]["download_name"]+".zip"
## For undirected grarph, we only store one directional information. This flag allows us to add inverse edge at pre-processing time
dataset_dict[name]["add_inverse_edge"] = False 
dataset_dict[name]["has_node_attr"] = False
dataset_dict[name]["has_edge_attr"] = False
dataset_dict[name]["split"] = "random"
dataset_dict[name]["additional node files"] = 'None'
dataset_dict[name]['additional edge files'] = 'edge_reltype'
dataset_dict[name]['is hetero'] = True

df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv("master.csv")
