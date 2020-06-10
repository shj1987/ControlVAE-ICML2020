#! /bin/sh

python main.py --dataset dsprites --seed 1 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --objective B --model B --batch_size 64 --z_dim 10 --max_iter 1.5e6 --gpu 0 \
    --C_stop_iter 1e5 --C_max 15 --gamma 100 --viz_name dsprites_B_gamma100_z10_c15_v1;
## second
python main.py --dataset dsprites --seed 2 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --objective B --model B --batch_size 64 --z_dim 10 --max_iter 1.5e6 --gpu 0 \
    --C_stop_iter 1e5 --C_max 15 --gamma 100 --viz_name dsprites_B_gamma100_z10_c15_v2;
## seed 3
python main.py --dataset dsprites --seed 3 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --objective B --model B --batch_size 64 --z_dim 10 --max_iter 1.5e6 --gpu 0 \
    --C_stop_iter 1e5 --C_max 15 --gamma 100 --viz_name dsprites_B_gamma100_z10_c15_v3;
## seed 4
python main.py --dataset dsprites --seed 4 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --objective B --model B --batch_size 64 --z_dim 10 --max_iter 1.5e6 --gpu 0 \
    --C_stop_iter 1e5 --C_max 15 --gamma 100 --viz_name dsprites_B_gamma100_z10_c15_v4;
## seed 5
python main.py --dataset dsprites --seed 5 --lr 5e-4 --beta1 0.9 --beta2 0.999 \
    --objective B --model B --batch_size 64 --z_dim 10 --max_iter 1.5e6 --gpu 0 \
    --C_stop_iter 1e5 --C_max 15 --gamma 100 --viz_name dsprites_B_gamma100_z10_c15_v5
