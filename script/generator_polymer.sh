#!/bin/bash

python train_generator --dataset_name polymer --max_num_epochs 100 --batch_size 64 --max_len 500 --test_num_samples 5000 --tags generator_polymer