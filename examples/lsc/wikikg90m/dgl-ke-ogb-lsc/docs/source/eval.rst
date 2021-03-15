Evaluation on Pre-Trained Embeddings
====================================

``dglke_eval`` reads the pre-trained embeddings and evaluates the quality of the embeddings with a link prediction task on the test set.

Arguments
---------
The command line provides the following arguments:

  - ``--model_name {TransE, TransE_l1, TransE_l2, TransR, RESCAL, DistMult, ComplEx, RotatE}``
    The models provided by DGL-KE.

  - ``--data_path DATA_PATH``
    The name of the knowledge graph stored under data_path. If it is one ofthe builtin knowledge grpahs such as FB15k, DGL-KE will automatically download the knowledge graph and keep it under data_path.

  - ``--dataset DATASET``     
    The name of the knowledge graph stored under data_path. If it is one ofthe builtin knowledge grpahs such as FB15k, DGL-KE will automatically download the knowledge graph and keep it under data_path.

  - ``--format FORMAT``
    The format of the dataset. For builtin knowledge graphs, the format is determined automatically. For users own knowledge graphs, it needs to be ``raw_udd_{htr}`` or ``udd_{htr}``. ``raw_udd_`` indicates that the user's data use **raw ID** for entities and relations and ``udd_`` indicates that the user's data uses **KGE ID**. ``{htr}`` indicates the location of the head entity, tail entity and relation in a triplet. For example, ``htr`` means the head entity is the first element in the triplet, the tail entity is the second element and the relation is the last element.

  - ``--data_files [DATA_FILES ...]``
    A list of data file names. This is used if users want to train KGE on their own datasets. If the format is *raw_udd_{htr}*, users need to provide *train_file* [*valid_file*] [*test_file*]. If the format is *udd_{htr}*, users need to provide *entity_file* *relation_file* *train_file* [*valid_file*] [*test_file*]. In both cases, *valid_file* and *test_file* are optional.

  - ``--delimiter DELIMITER``
    Delimiter used in data files. Note all files should use the same delimiter.

  - ``--model_path MODEL_PATH``
    The place where models are saved.

  - ``--batch_size_eval BATCH_SIZE_EVAL``
    Batch size used for eval and test

  - ``--neg_sample_size_eval NEG_SAMPLE_SIZE_EVAL``
    Negative sampling size for testing

  - ``--neg_deg_sample_eval``
    Negative sampling proportional to vertex degree for testing.

  - ``--hidden_dim HIDDEN_DIM``
    Hidden dim used by relation and entity

  - ``-g GAMMA`` or ``--gamma GAMMA``
    The margin value in the score function. It is used by *TransX* and *RotatE*.

  - ``--eval_percent EVAL_PERCENT``
    Randomly sample some percentage of edges for evaluation.

  - ``--no_eval_filter`` 
    Disable filter positive edges from randomly constructed negative edges for evaluation.

  - ``--gpu [GPU ...]``
    A list of gpu ids, e.g. 0 1 2 4

  - ``--mix_cpu_gpu``         
    Training a knowledge graph embedding model with both CPUs and GPUs.The embeddings are stored in CPU memory and the training is performed in GPUs.This is usually used for training a large knowledge graph embeddings. 

  - ``-de`` or ``--double_ent``
    Double entitiy dim for complex number It is used by *RotatE*.

  - ``-dr`` or ``--double_rel``
    Double relation dim for complex number.

  - ``--num_proc NUM_PROC`` 
    The number of processes to train the model in parallel.In multi-GPU training, the number of processes by default is set to match the number of GPUs. If set explicitly, the number of processes needs to be divisible by the number of GPUs.

  - ``--num_thread NUM_THREAD``
    The number of CPU threads to train the model in each process. This argument is used for multi-processing training.


Examples
--------


The following command evaluates the pre-trained KG embedding on multi-cores::

    dglke_eval --model_name TransE_l2 --dataset FB15k --hidden_dim 400 --gamma 19.9 --batch_size_eval 16 \
    --num_thread 1 --num_proc 8 --model_path ~/my_task/ckpts/TransE_l2_FB15k_0/

We can also use GPUs in our evaluation tasks::

    dglke_eval --model_name TransE_l2 --dataset FB15k --hidden_dim 400 --gamma 19.9 --batch_size_eval 16 \
    --gpu 0 1 2 3 4 5 6 7 --model_path ~/my_task/ckpts/TransE_l2_FB15k_0/

