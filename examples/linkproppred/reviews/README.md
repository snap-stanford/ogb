# ogbl-reviews

This repository includes the following example scripts:

* **[Matrix Factorization](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/reviews/mf.py)**: Full-batch Matrix Factorization training. To train with product features, please use the `--use_node_features` argument.

By default, examples will use the `ogbl-reviews-groc` dataset.
To train the models on the `ogbl-reviews-book` dataset, use the `--suffix=book` argument.

## Training & Evaluation

```
# Run with default config
python mf.py

# Run with custom config
python mf.py --hidden_channels=128
```
