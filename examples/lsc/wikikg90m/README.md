# Baseline code for Wikikg90m

Please refer to our OGB-LSC paper (coming soon) for the detailed setting.

## Installation requirements
```
ogb>=1.3.0
torch>=1.7.0
dgl==0.4.3
```
Also install the dgl-ke-ogb-lsc by `cd dgl-ke-ogb-lsc/python` and `pip install -e .`

## Key commandline arguments
- `train_mode`: Choose within [`emb`, `roberta` and `both`], which corresponds to the `x-shallow`, `x-roberta`, `x-concat` respectively (details in the paper).
- `data_path`: Directory that downloads and stores the dataset.

## Baseline models
- TransE-Shallow [1]
- TransE-RoBERTa [1,3]
- TransE-Concat [1,3]
- ComplEx-Shallow [2]
- ComplEx-RoBERTa [2,3]
- ComplEx-Concat [2,3]

Example scripts are in `examples.sh`


## Performance

| Model              |Valid MRR  | Test MRR*   | \#Parameters    | Hardware |
|:------------------ |:--------------   |:---------------| --------------:|----------|
| TransE-Shallow     | 0.7559 | 0.7412 | 17.4B  | Tesla P100 (16GB GPU) |
| ComplEx-Shallow    | - | - | 17.4B  | Tesla P100 (16GB GPU) |
| TransE-RoBERTa     | 0.6039 | 0.6288 | 0.3M   | Tesla P100 (16GB GPU) |
| ComplEx-RoBERTa    | - | - | 0.3M   | Tesla P100 (16GB GPU) |
| TransE-Concat      | 0.8494 | 0.8548 | 17.4B  | Tesla P100 (16GB GPU) |
| ComplEx-Concat     | - | - | 17.4B  | Tesla P100 (16GB GPU) |

\* Test MRR is evaluated on the **hidden test set.**

## References
[1] Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. NeurIPS 2013

[2] Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex embeddings for simple link prediction. ICML 2016

[3] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A robustly optimized bert pretraining approach.
