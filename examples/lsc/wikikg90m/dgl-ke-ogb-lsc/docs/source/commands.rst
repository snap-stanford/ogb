DGL-KE Command Lines
====================

DGL-KE provides a set of command line tools to train knowledge graph embeddings and make prediction
with the embeddings easily. 

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   format_kg
   format_out
   train
   partition
   dist_train
   eval
   predict
   emb_sim

Commands for Training
---------------------

DGL-KE provides commands to support training on CPUs, GPUs in a single machine and a cluster of machines.

``dglke_train`` trains KG embeddings on CPUs or GPUs in a single machine and saves the trained node embeddings and relation embeddings on disks.

``dglke_dist_train`` trains knowledge graph embeddings on a cluster of machines. This command launches a set of processes to perform distributed training automatically.

To support distributed training, DGL-KE provides a command to partition a knowledge graph before training.

``dglke_partition`` partitions the given knowledge graph into ``N`` parts by the METIS partition algorithm. Different partitions will be stored on different machines in distributed training. You can find more details about the METIS partition algorithm in this `link`__.

.. __: http://glaros.dtc.umn.edu/gkhome/metis/metis/overview

In addition, DGL-kE provides a command to evaluate the quality of pre-trained embeddings.

``dglke_eval`` reads the pre-trained embeddings and evaluates the quality of the embeddings with a link prediction task on the test set.

Commands for Inference
----------------------

DGL-KE supports two types of inference tasks using pretained embeddings (We recommand using DGL-KE to generate these embedding).

  * **Predicting entities/relations in a triplet** Given entities and/or relations, predict which entities or relations are likely to connect with the existing entities for given relations. For example, given a head entity and a relation, predict which entities are likely to connect to the head entity via the given relation.
  * **Finding similar embeddings** Given entity/relation embeddings, find the most similar entity/relation embeddings for some pre-defined similarity functions.

The ranking result will be automatically stored in the output file (result.tsv by default) using the tsv format. DGL-KE provides two commands for the inference tasks:

``dglke_predict`` predicts missing entities/relations in triplets using the pre-trained embeddings.

``dglke_emb_sim`` computes similarity scores on the entity embeddings or relation embeddings.
