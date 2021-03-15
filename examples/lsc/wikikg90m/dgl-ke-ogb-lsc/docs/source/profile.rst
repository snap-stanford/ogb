Profile DGL-KE
--------------

This document is mainly for developing the DGL-KE models and accelerating their training.

To analyze MXNet version of KE models, please enable `MXNet_PROFILER` environment variable when running the training job::

    MXNET_PROFILER=1 dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
    --gamma 19.9 --lr 0.25 --max_step 3000 --log_interval 100 --batch_size_eval 16 --test -adv \
    --regularization_coef 1.00E-09 --num_thread 1 --num_proc 8


