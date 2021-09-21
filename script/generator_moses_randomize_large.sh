#!/bin/bash

python train_generator2.py \
--dataset_name moses \
--randomize \
--num_layers 6 \
--batch_size 64 \
--max_epochs 50 \
--test_num_samples 30000 \
--tag generator_moses_randomize_large