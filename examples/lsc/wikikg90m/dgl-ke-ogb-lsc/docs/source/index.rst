.. dglke documentation master file, created by
   sphinx-quickstart on Wed Apr  1 12:56:21 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DGL-KE Documentation
========================================

Knowledge graphs (KGs) are data structures that store information about different entities (nodes) and their relations (edges). A common approach of using KGs in various machine learning tasks is to compute knowledge graph embeddings. DGL-KE is a high performance, easy-to-use, and scalable package for learning large-scale knowledge graph embeddings. The package is implemented on the top of Deep Graph Library (`DGL`__) and developers can run DGL-KE on CPU machine, GPU machine, as well as clusters with a set of popular models, including `TransE`__, `TransR`__, `RESCAL`__, `DistMult`__, `ComplEx`__, and `RotatE`__.

.. __: https://www.dgl.ai/
.. __: https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
.. __: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9571
.. __: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.2015&rep=rep1&type=pdf
.. __: https://arxiv.org/abs/1412.6575
.. __: http://proceedings.mlr.press/v48/trouillon16.pdf
.. __: https://arxiv.org/pdf/1902.10197.pdf

.. image:: https://data.dgl.ai/asset/image/ke/dgl_ke_arch.png
    :width: 650  

Performance and Scalability
----------------------------------------

DGL-KE is designed for learning at scale. It introduces various novel optimizations that accelerate training on knowledge graphs with millions of nodes and billions of edges. Our benchmark on knowledge graphs consisting of over *86M* nodes and *338M* edges shows that DGL-KE can compute embeddings in 100 minutes on an EC2 instance with 8 GPUs and 30 minutes on an EC2 cluster with 4 machines (48 cores/machine). These results represent a *2×∼5×* speedup over the best competing approaches.

*DGL-KE vs Graphvite*

.. image:: https://data.dgl.ai/asset/image/ke/vs-gv-fb15k.png
    :width: 650 

*DGL-KE vs Pytorch-Biggraph*

.. image:: https://data.dgl.ai/asset/image/ke/vs-pbg-fb.png
    :width: 650 

Get started with DGL-KE!
----------------------------------------

.. toctree::
   :maxdepth: 2

   install
   kg
   commands
   train_user_data
   benchmarks
   profile
