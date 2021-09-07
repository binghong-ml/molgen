#!/bin/bash

python train_generator.py \
--dataset_name moses \
--max_epochs 10 \
--batch_size 256 \
--max_len 250 \
--test_num_samples 30000 \
--use_linedistance \
--tag generator_moses_linedistance
