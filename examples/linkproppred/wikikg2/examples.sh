#!/bin/bash


# 100 dimension
python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --model TransE -n 128 -b 512 -d 100 -g 30 -a 1.0 -adv \
  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32

python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --model DistMult -n 128 -b 512 -d 100 -g 500 -a 1.0 -adv \
  -lr 0.001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 -r 0.000002

python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --model RotatE -n 128 -b 512 -d 50 -g 5 -a 1.0 -adv \
  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 -de

python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --model ComplEx -n 128 -b 512 -d 50 -g 500 -a 1.0 -adv \
  -lr 0.001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 -de -dr -r 0.000002

# 600 dimension
python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --model TransE -n 128 -b 512 -d 500 -g 30 -a 1.0 -adv \
  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32

python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --model DistMult -n 128 -b 512 -d 500 -g 500 -a 1.0 -adv \
  -lr 0.001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 -r 0.000002

python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --model RotatE -n 128 -b 512 -d 250 -g 5 -a 1.0 -adv \
  -lr 0.0001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 -de

python run.py --do_train --cuda --do_valid --do_test --evaluate_train \
  --model ComplEx -n 128 -b 512 -d 250 -g 500 -a 1.0 -adv \
  -lr 0.001 --max_steps 200000 --cpu_num 2 --test_batch_size 32 -de -dr -r 0.000002
