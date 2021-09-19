#!/bin/bash

python train_generator.py \
--dataset_name moses \
--max_epochs 20 \
--num_layers 6 \
--batch_size 128 \
--test_num_samples 30000 \
--tag generator_moses0