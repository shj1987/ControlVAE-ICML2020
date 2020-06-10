#! /bin/sh

python3 main.py --train True --dataset celeba --seed 1 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 100 --z_dim 500 --max_iter 1.2e6 \
    --beta 1 --viz_name celeba_pid_z500_kl170-v1  --is_PID True --KL_loss 170 --image_size 128 ;

