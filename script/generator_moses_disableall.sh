#!/bin/bash

python train_generator.py \
--dataset_name moses \
--disable_branchidx \
--disable_loc \
--disable_edgelogit \
--tag disableall \
--max_epochs 20 \
--test_num_samples 30000 \
--tag generator_moses_disableall