Predict entities/relations in triplets
======================================
`dglke_predict` predicts missing entities or relations in a triplet. Blow shows an example that predicts top 5 most likely destination entities for every given source node and relation::

   src  rel  dst   score
    1    0    12   -5.11393
    1    0    18   -6.10925
    1    0    13   -6.66778
    1    0    17   -6.81532
    1    0    19   -6.83329
    2    0    17   -5.09325
    2    0    18   -5.42972
    2    0    20   -5.61894
    2    0    12   -5.75848
    2    0    14   -5.94183

Currently, it supports six models: TransE_l1, TransE_l2, RESCAL, DistMult, ComplEx, and RotatE.

Arguments
---------

Four arguments are required to provide basic information for predicting missing entities or relations:

  * ``--model_path``, The path containing the pretrained model, including the embedding files (.npy) and a config.json containing the configuration of training the model.
  * ``--format``, The format of the input data, specified in ``h_r_t``. Ideally, user should provides three files, one for head entities, one for relations and one for tail entities. But we also allow users to use *\** to represent *all* of the entities or relations. For example, ``h_r_*`` requires users to provide files containing head entities and relation entities and use all entities as tail entities; ``*_*_t`` requires users to provide a single file containing tail entities and use all entities as head entities and all relations. The supported formats include ``h_r_t``, ``h_r_*``, ``h_*_t``, ``*_r_t``, ``h_*_*``, ``*_r_*``, ``*_*_t``.
  * ``--data_files`` A list of data file names. This is used to provide necessary files containing the input data according to the format, e.g., for ``h_r_t``, the three input files are required and they contain a list of head entities, a list of relations and a list of tail entities. For ``h_*_t``, two files are required, which contain a list of head entities and a list of tail entities.
  * ``--raw_data``, A flag indicates whether the input data specified by --data_files use the raw Ids or KGE Ids. If True, the input data uses Raw IDs and the command translates IDs according to ID mapping. If False, the data use KGE IDs. Default False.

Task related arguments:

  * ``--exec_mode``, How to calculate scores for triplets and calculate topK. Default 'all'.

    * ``triplet_wise``: head, relation and tail lists have the same length N, and we calculate the similarity triplet by triplet: result = topK([score(h_i, r_i, t_i) for i in N]), the result shape will be (K,).
    * ``all``: three lists of head, relation and tail ids are provided as H, R and T, and we calculate all possible combinations of all triplets (h_i, r_j, t_k): result = topK([[[score(h_i, r_j, t_k) for each h_i in H] for each r_j in R] for each t_k in T]), and find top K from the triplets
    * ``batch_head``: three lists of head, relation and tail ids are provided as H, R and T, and we calculate topK for each element in head: result = topK([[score(h_i, r_j, t_k) for each r_j in R] for each t_k in T]) for each h_i in H. It returns (sizeof(H) * K) triplets.
    * ``batch_rel``: three lists of head, relation and tail ids are provided as H, R and T, and we calculate topK for each element in relation: result = topK([[score(h_i, r_j, t_k) for each h_i in H] for each t_k in T]) for each r_j in R. It returns (sizeof(R) * K) triplets.
    * ``batch_tail``: three lists of head, relation and tail ids are provided as H, R and T, and we calculate topK for each element in tail: result = topK([[score(h_i, r_j, t_k) for each h_i in H] for each r_j in R]) for each t_k in T. It returns (sizeof(T) * K) triplets.

  * ``--topk``, How many results are returned. Default: 10.
  * ``--score_func``, What kind of score is used in ranking. Currently, we support two functions: ``none`` (score = $x$) and ``logsigmoid`` ($score = log(sigmoid(x))$). Default: 'none'.
  * ``--gpu``, GPU device to use in inference. Default: -1 (CPU)

Input/Output related arguments:

  * ``--output``, the output file to store the result. By default it is stored in result.tsv
  * ``--entity_mfile``, The entity ID mapping file. Required if Raw ID is used.
  * ``--rel_mfile``, The relation ID mapping file. Required if Raw ID is used.

Examples
--------

The following command predicts the K most likely relations and tail entities for each head entity in the list using a pretrained TransE_l2 model (--exec_mode ‘batch_head’). In this example, the candidate relations and the candidate tail entities are given by the user.::

    # Using PyTorch Backend
    dglke_predict --model_path ckpts/TransE_l2_wn18_0/ --format 'h_r_t' --data_files head.list rel.list tail.list --score_func logsigmoid --topK 5 --exec_mode 'batch_head'

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_predict --model_path ckpts/TransE_l2_wn18_0/ --format 'h_r_t' --data_files head.list rel.list tail.list --score_func logsigmoid --topK 5  --exec_mode 'batch_head'

The output is as::

    src  rel  dst  score
    1    0    12   -5.11393
    1    0    18   -6.10925
    1    0    13   -6.66778
    1    0    17   -6.81532
    1    0    19   -6.83329
    2    0    17   -5.09325
    2    0    18   -5.42972
    2    0    20   -5.61894
    2    0    12   -5.75848
    2    0    14   -5.94183
    ...

The following command finds the most likely combinations of head entities, relations and tail entities from the input lists using a pretrained DistMult model::

    # Using PyTorch Backend
    dglke_predict --model_path ckpts/DistMult_wn18_0/ --format 'h_r_t' --data_files head.list rel.list tail.list --score_func none --topK 5

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_predict --model_path ckpts/DistMult_wn18_0/ --format 'h_r_t' --data_files head.list rel.list tail.list --score_func none --topK 5

The output is as::

    src  rel  dst  score
    6    0    15   -2.39380
    8    0    14   -2.65297
    2    0    14   -2.67331
    9    0    18   -2.86985
    8    0    20   -2.89651

The following command finds the most likely combinations of head entities, relations and tail entities from the input lists using a pretrained TransE_l2 model and uses Raw ID (turn on --raw_data)::

    # Using PyTorch Backend
    dglke_predict --model_path ckpts/TransE_l2_wn18_0/ --format 'h_r_t' --data_files raw_head.list raw_rel.list raw_tail.list --topK 5 --raw_data --entity_mfile data/wn18/entities.dict --rel_mfile data/wn18/relations.dict

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_predict --model_path ckpts/TransE_l2_wn18_0/ --format 'h_r_t' --data_files raw_head.list raw_rel.list raw_tail.list --topK 5 --raw_data --entity_mfile data/wn18/entities.dict --rel_mfile data/wn18/relations.dict

The output is as::

    head      rel                           tail      score
    08847694  _derivationally_related_form  09440400  -7.41088
    08847694  _hyponym                      09440400  -8.99562
    02537319  _derivationally_related_form  01490112  -9.08666
    02537319  _hyponym                      01490112  -9.44877
    00083809  _derivationally_related_form  05940414  -9.88155
