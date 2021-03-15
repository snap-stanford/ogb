Find similar embeddings
=======================
`dglke_emb_sim` finds the most similar entity/relation embeddings for some pre-defined similarity functions given a set of entities or relations. An example of the output for top5 similar entities are as follows::

    left     right    score
    0        0        0.99999
    0        18470    0.91855
    0        2105     0.89916
    0        13605    0.83187
    0        36762    0.76978

Currently we support five different similarity functions: cosine, l2 distance, l1 distance, dot product and extended jaccard.

Arguments
---------

Four arguments are required to provide basic information for finding similar embeddings:

  * ``--emb_file``, The numpy file that contains the embeddings of all entities/relations in a knowledge graph.
  * ``--format``, The format of the input objects (entities/relations).

    * ``l_r``: two list of objects are provided as left objects and right objects.
    * ``l_*``: one list of objects is provided as left objects and all objects in emb\_file are right objects. This is to find most similar objects to the ones on the left.
    * ``*_r``: one list of objects is provided as right objects list and treat all objects in emb\_file as left objects.
    * ``*``: all objects in the emb\_file are both left objects and right objects. The option finds the most similar objects in the graph.

  * ``--data_files`` A list of data file names. It provides necessary files containing the requried data according to the format, e.g., for ``l_r``, two files are required as left_data and right_data, while for ``l_*``, one file is required as left_data, and for ``*`` this argument will be omited.
  * ``--raw_data``, A flag indicates whether the data in data_files are raw IDs or KGE IDs. If True, the data are the Raw IDs and the command will map the raw IDs to KGE Ids automatically using the ID mapping file provided through ``--mfile``. If False, the data are KGE IDs. Default: False.

Task related arguments:

  * ``--exec_mode``, Indicate how to calculate scores for element pairs and calculate topK. Default: 'all'

    * ``pairwise``: The same number (N) of left and right objects are provided. It calculates the similarity pair by pair: result = topK([score(l_i, r_i) for i in N]) and output the K most similar pairs.
    * ``all``: both left objects and right objects are provided as L and R. It calculates similarity scores of all possible combinations of (l_i, r_j): result = topK([[score(l_i, rj) for l_i in L] for r_j in R]), and outputs the K most similar pairs.
    * ``batch_left``: left objects and right objects are provided as L and R. It finds the K most similar objects from the right objects for each object in L: result = topK([score(l_i, r_j) for r_j in R]) for l_j in L. It outputs (len(L) * K) most similar pairs.

  * ``--topk``, How many results are returned. Default: 10.
  * ``--sim_func``, the function to define the similarity score between a pair of objects. It support five functions. Default: cosine
  
    * **cosine**: use cosine similarity; score = $\frac{x \cdot y}{||x||_2||y||_2}$
    * **l2**: use l2 similarity; score = $-||x - y||_2$
    * **l1**: use l1 similarity; score = $-||x - y||_1$
    * **dot**: use dot product similarity; score = $x \cdot y$
    * **ext_jaccard**: use extended jaccard similarity. score = $\frac{x \cdot y}{||x||_{2}^{2} + ||y||_{2}^{2} - x \cdot y}$

  * ``--gpu``, GPU device to use in inference. Default: -1 (CPU).

Input/Output related arguments:

  * ``--output``, the output file that stores the result. By default it is stored in result.tsv.
  * ``--mfile``, The ID mapping file.

Examples
--------

The following command finds similar entities based on cosine distance::

    # Using PyTorch Backend
    dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list  --topK 5

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list --topK 5

The output is as::

    left    right   score
    6       15      0.55512
    1       12      0.33153
    7       20      0.27706
    7       19      0.25631
    7       13      0.21372

The following command finds topK most similar entities for each element on the left using l2 distance (--exec_mode batch_left)::

    # Using PyTorch Backend
    dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_*' --data_files head.list --sim_func l2 --topK 5 --exec_mode 'batch_left'

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_*' --data_files head.list --sim_func l2 --topK 5 --exec_mode 'batch_left'

The output is as::

    left    right   score
    0       0       0.0
    0       18470   3.1008
    0       24408   3.1466
    0       2105    3.3411
    0       13605   4.1587
    1       1       0.0
    1       26231   4.9025
    1       2617    5.0204
    1       12672   5.2221
    1       38633   5.3221
    ...

The following command finds similar relations using cosine distance and use Raw ID (turn on --raw_data)::

    # Using PyTorch Backend
    dglke_emb_sim --mfile data/wn18/relations.dict --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_relation.npy  --format 'l_*' --data_files raw_rel.list --topK 5 --raw_data

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_emb_sim --mfile data/wn18/relations.dict --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_relation.npy  --format 'l_*' --data_files raw_rel.list --topK 5 --raw_data

The output is as::

    left                          right                           score
    _hyponym                      _hyponym                        0.99999
    _derivationally_related_form  _derivationally_related_form    0.99999
    _hyponym                      _also_see                       0.58408
    _hyponym                      _member_of_domain_topic         0.44027
    _hyponym                      _member_of_domain_region        0.30975
