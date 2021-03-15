# TransE-shallow
dglke_train --model_name TransE_l2 --dataset wikikg90m --batch_size 400 \
--neg_sample_size 100 --hidden_dim 200 --gamma 10 --lr 0.1 --regularization_coef 1e-9 \
--batch_size_eval 50 --valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 --max_step 1000000 --neg_sample_size_eval 1000 \
--log_interval 1000 --async_update --force_sync_interval 10000 --no_save_emb --print_on_screen --train_mode emb \
--eval_interval 50000

# TransE-roberta
dglke_train --model_name TransE_l2 --dataset wikikg90m --batch_size 400 \
--neg_sample_size 100 --hidden_dim 200 --gamma 10 --lr 0.1 --regularization_coef 1e-9 \
--batch_size_eval 50 --valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 --max_step 1000000 --neg_sample_size_eval 1000 \
--log_interval 1000 --async_update --force_sync_interval 10000 --no_save_emb --print_on_screen --train_mode roberta \
--eval_interval 50000

# TransE-concat
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --model_name TransE_l2 --dataset wikikg90m --batch_size 400 \
--neg_sample_size 100 --hidden_dim 200 --gamma 10 --lr 0.1 --regularization_coef 1e-9 \
--batch_size_eval 50 --valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 --max_step 1000000 --neg_sample_size_eval 1000 \
--log_interval 1000 --async_update --force_sync_interval 50000 --no_save_emb --print_on_screen --train_mode both \
--eval_interval 50000

# ComplEx-shallow
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --model_name ComplEx --dataset wikikg90m --batch_size 400 \
--neg_sample_size 100 --hidden_dim 100 --gamma 8 --lr 0.01 --regularization_coef 2e-6 \
--batch_size_eval 50 --valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 --max_step 1000000 --neg_sample_size_eval 1000 \
--log_interval 1000 --async_update --force_sync_interval 50000 --no_save_emb --print_on_screen --train_mode emb \
--eval_interval 50000 -de -dr

# ComplEx-roberta
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --model_name ComplEx --dataset wikikg90m --batch_size 400 \
--neg_sample_size 100 --hidden_dim 100 --gamma 100 --lr 0.1 --regularization_coef 1e-9 \
--batch_size_eval 50 --valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 --max_step 1000000 --neg_sample_size_eval 1000 \
--log_interval 1000 --async_update --force_sync_interval 10000 --no_save_emb --print_on_screen --train_mode roberta \
--eval_interval 50000 -de -dr

# ComplEx-concat
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --model_name ComplEx --dataset wikikg90m --batch_size 400 \
--neg_sample_size 100 --hidden_dim 100 --gamma 3 --lr 0.1 --regularization_coef 1e-9 \
--batch_size_eval 50 --valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 --max_step 1000000 --neg_sample_size_eval 1000 \
--log_interval 1000 --async_update --force_sync_interval 50000 --no_save_emb --print_on_screen --train_mode both \
--eval_interval 50000 -de -dr
