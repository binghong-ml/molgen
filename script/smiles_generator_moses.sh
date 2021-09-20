#!/bin/bash

python train_smiles_generator.py --dataset_name moses --max_epochs 20 --test_num_samples 30000 --tag smiles_generator_moses