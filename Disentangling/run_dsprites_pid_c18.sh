#! /bin/sh


python3 main.py --train True --dataset dsprites --seed 1 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --C_stop_iter 5e5 --step_val 0.15 --gpu 0 \
    --viz_name dsprites_Dynamic_pid_c18 --C_max 18;


