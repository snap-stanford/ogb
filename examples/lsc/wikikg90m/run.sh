# TransE-shallow
dglke_train --model_name TransE_l2 \
--hidden_dim 200 --gamma 10 --lr 0.1 --regularization_coef 1e-9 \
--valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 \
--async_update --force_sync_interval 10000 --no_save_emb \
--print_on_screen --encoder_model_name shallow --save_path $SAVE_PATH


# TransE-roberta
dglke_train --model_name TransE_l2 \
--hidden_dim 200 --gamma 10 --lr 0.1 --regularization_coef 1e-9 \
--valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 \
--async_update --force_sync_interval 10000 --no_save_emb \
--print_on_screen --encoder_model_name roberta --save_path $SAVE_PATH


# TransE-concat
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --model_name TransE_l2 \
--hidden_dim 200 --gamma 10 --lr 0.1 --regularization_coef 1e-9 \
--valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 \
--async_update --force_sync_interval 50000 --no_save_emb \
--print_on_screen --encoder_model_name concat --save_path $SAVE_PATH


# ComplEx-shallow
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --model_name ComplEx \
--hidden_dim 100 --gamma 8 --lr 0.01 --regularization_coef 2e-6 \
--valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 \
--async_update --force_sync_interval 50000 --no_save_emb \
--print_on_screen --encoder_model_name shallow -de -dr --save_path $SAVE_PATH

# ComplEx-roberta
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --model_name ComplEx \
--hidden_dim 100 --gamma 100 --lr 0.1 --regularization_coef 1e-9 \
--valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 \
--async_update --force_sync_interval 10000 --no_save_emb \
--print_on_screen --encoder_model_name roberta -de -dr --save_path $SAVE_PATH

# ComplEx-concat
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --model_name ComplEx \
--hidden_dim 100 --gamma 3 --lr 0.1 --regularization_coef 1e-9 \
--valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 \
--async_update --force_sync_interval 50000 --no_save_emb \
--print_on_screen --encoder_model_name concat -de -dr --save_path $SAVE_PATH
