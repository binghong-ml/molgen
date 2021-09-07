#!/bin/bash

python train_generator.py --dataset_name polymer --max_epochs 100 --batch_size 64 --max_len 500 --test_num_samples 5000 --tag generator_polymer