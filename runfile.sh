#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0, python ./main.py --name git-commit-ISIT --config_name awgn --P  0.0100 --C 0.0050 &
#CUDA_VISIBLE_DEVICES=0, python ./main.py --name git-commit-ISIT --config_name awgn --P  0.0316 --C 0.0156 &
#CUDA_VISIBLE_DEVICES=0, python ./main.py --name git-commit-ISIT --config_name awgn --P  0.1000 --C 0.0477 &
#CUDA_VISIBLE_DEVICES=0, python ./main.py --name git-commit-ISIT --config_name awgn --P  0.3160 --C 0.1374 &
CUDA_VISIBLE_DEVICES=0, python ./main.py --name git-commit-ISIT --config_name awgn --P  1.0000 --C 0.3466 &
CUDA_VISIBLE_DEVICES=0, python ./main.py --name git-commit-ISIT --config_name awgn --P  3.1620 --C 0.7130 &
CUDA_VISIBLE_DEVICES=1, python ./main.py --name git-commit-ISIT --config_name awgn --P  10.000 --C 1.1989 &
CUDA_VISIBLE_DEVICES=1, python ./main.py --name git-commit-ISIT --config_name awgn --P  31.620 --C 1.7425 &
#CUDA_VISIBLE_DEVICES=3, python ./main.py --name git-commit-ISIT --config_name arma_ff --P  0.0100 --C 0.0161 &
#CUDA_VISIBLE_DEVICES=3, python ./main.py --name git-commit-ISIT --config_name arma_ff --P  0.0316 --C 0.0423 &
#CUDA_VISIBLE_DEVICES=3, python ./main.py --name git-commit-ISIT --config_name arma_ff --P  0.1000 --C 0.0996 &
#CUDA_VISIBLE_DEVICES=3, python ./main.py --name git-commit-ISIT --config_name arma_ff --P  0.3160 --C 0.2100 &
CUDA_VISIBLE_DEVICES=3, python ./main.py --name git-commit-ISIT --config_name arma_ff --P  1.0000 --C 0.4054 &
CUDA_VISIBLE_DEVICES=3, python ./main.py --name git-commit-ISIT --config_name arma_ff --P  3.1620 --C 0.7420 &
CUDA_VISIBLE_DEVICES=5, python ./main.py --name git-commit-ISIT --config_name arma_ff --P  10.000 --C 1.2100 &
CUDA_VISIBLE_DEVICES=5, python ./main.py --name git-commit-ISIT --config_name arma_ff --P  31.620 --C 1.7460 &
#CUDA_VISIBLE_DEVICES=5, python ./main.py --name git-commit-ISIT --config_name arma_fb --P  0.0100 --C 0.0189 &
#CUDA_VISIBLE_DEVICES=6, python ./main.py --name git-commit-ISIT --config_name arma_fb --P  0.0316 --C 0.0541 &
#CUDA_VISIBLE_DEVICES=6, python ./main.py --name git-commit-ISIT --config_name arma_fb --P  0.1000 --C 0.1367 &
#CUDA_VISIBLE_DEVICES=6, python ./main.py --name git-commit-ISIT --config_name arma_fb --P  0.3160 --C 0.2947 &
CUDA_VISIBLE_DEVICES=6, python ./main.py --name git-commit-ISIT --config_name arma_fb --P  1.0000 --C 0.5462 &
CUDA_VISIBLE_DEVICES=6, python ./main.py --name git-commit-ISIT --config_name arma_fb --P  3.1620 --C 0.8954 &
CUDA_VISIBLE_DEVICES=2, python ./main.py --name git-commit-ISIT --config_name arma_fb --P  10.000 --C 1.3295 &
CUDA_VISIBLE_DEVICES=7, python ./main.py --name git-commit-ISIT --config_name arma_fb --P  31.620 --C 1.8242 &

