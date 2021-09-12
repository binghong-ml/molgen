#!/bin/bash

python train_generator.py \
--dataset_name zinc \
--max_epochs 100 \
--batch_size 256 \
--max_len 250 \
--test_num_samples 30000 \
--tag generator_zinc
