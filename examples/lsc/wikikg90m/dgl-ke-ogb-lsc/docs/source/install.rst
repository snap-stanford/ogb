Installation Guide
==================


This topic explains how to install DGL-KE. We recommend installing DGL-KE by using ``pip`` and from the source.

System requirements
-------------------

DGL-KE works with the following operating systems:

- Ubuntu 16.04 or higher version
- macOS x

DGL-KE requires Python version 3.5 or later. Python 3.4 or earlier is not tested. Python 2 support is coming.

DGL-KE supports multiple tensor libraries as backends, e.g., PyTorch and MXNet. For requirements on backends and how to select one, see Working with different backends. As a demo, we install Pytorch using ``pip``::

    sudo pip3 install torch


Install DGL
-----------

DGL-KE is implemented on the top of DGL (0.4.3 version). You can install DGL using pip::

    sudo pip3 install dgl==0.4.3


Install DGL-KE 
--------------

After installing DGL, you can install DGL-KE. The fastest way to install DGL-KE is by using pip::

    sudo pip3 install dglke

or you can install DGL-KE from source::

    git clone https://github.com/awslabs/dgl-ke.git
    cd dgl-ke/python
    sudo python3 setup.py install


Have a Quick Test
-----------------

Once you install DGL-KE successfully, you can test it by the following command::

    # create a new workspace
    mkdir my_task && cd my_task 
    # Train transE model on FB15k dataset
    DGLBACKEND=pytorch dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 \
    --neg_sample_size 200 --hidden_dim 400 --gamma 19.9 --lr 0.25 --max_step 500 --log_interval 100 \
    --batch_size_eval 16 -adv --regularization_coef 1.00E-09 --test --num_thread 1 --num_proc 8

This command will download the ``FB15k`` dataset, train the ``transE`` model on that, and save the trained embeddings into the file. You could see the following output at the end::

    -------------- Test result --------------
    Test average MRR : 0.47221913961451095
    Test average MR : 58.68289854581774
    Test average HITS@1 : 0.2784276548560207
    Test average HITS@3 : 0.6244265375564998
    Test average HITS@10 : 0.7726295474936941
    -----------------------------------------
