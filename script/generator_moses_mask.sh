#!/bin/bash

python train_generator2.py \
--use_valence_mask \
--dataset_name moses \
--max_epochs 50 \
--test_num_samples 30000 \
--tag generator_moses_mask