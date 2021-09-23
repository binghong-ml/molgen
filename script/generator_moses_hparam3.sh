#!/bin/bash

python train_generator.py \
--lr 2e-4 \
--num_layers 6 \
--input_dropout 0.0 \
--randomize \
--dataset_name moses \
--max_epochs 100 \
--check_sample_every_n_epoch 2 \
--num_samples 30000 \
--eval_moses \
--tag generator_moses_hparam3