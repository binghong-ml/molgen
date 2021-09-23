#!/bin/bash

python train_generator.py \
--lr 2e-4 \
--num_layers 3 \
--input_dropout 0.0 \
--dataset_name moses \
--max_epochs 100 \
--check_sample_every_n_epoch 5 \
--num_samples 30000 \
--eval_moses \
--tag generator_moses_hparam1