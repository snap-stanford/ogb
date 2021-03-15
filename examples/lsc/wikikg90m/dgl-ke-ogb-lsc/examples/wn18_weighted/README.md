# Weighted WN18 Example
This example shows how to train a knowledge graph with weighted edges (each edge has an importance score)

## How to get data
```
>>> wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/KGE_Examples/wn18_weighted_edge/wn18_weighted.tgz
>>> tar -zxf wn18_weighted.tgz
>>> ls wn18_weighted
README  entities.dict  relations.dict  test_weight.txt  train_weight.txt  valid_weight.txt
```

## How to train
```
dglke_train --model_name TransE_l1 --dataset wn18-weight --format raw_udd_hrt --data_files train_weight.txt valid_weight.txt test_weight.txt --data_path ./data/wn18_weighted/ --batch_size 2048 --log_interval 1000 --neg_sample_size 128 --regularization_coef 2e-07 --hidden_dim 512 --gamma 12.0 --lr 0.007 --batch_size_eval 16 --test -adv --gpu 0 --max_step 32000 --has_edge_importance
```
