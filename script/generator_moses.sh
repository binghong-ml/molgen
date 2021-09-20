#!/bin/bash

python train_generator.py \
--dataset_name moses \
--max_epochs 50 \
--test_num_samples 30000 \
--resume_from_checkpoint_path ../resource/checkpoint/generator_moses/epoch\=19-step\=123819.ckpt \
--tag generator_moses