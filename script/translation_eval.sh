#!/bin/bash

python train_translation.py \
--max_epochs 0 \
--debug \
--dataset_name logp04 \
--load_checkpoint_path ../resource/checkpoint/translation_logp04/epoch\=62-step\=97334.ckpt \
--checkpoint_dir ../resource/checkpoint/translation_logp04

#python train_translation.py \
#--max_epochs 0 \
#--debug \
#--dataset_name logp06 \
#--load_checkpoint_path ../resource/checkpoint/translation_logp06/epoch\=66-step\=78456.ckpt \
#--checkpoint_dir ../resource/checkpoint/translation_logp06

#python train_translation.py \
#--max_epochs 0 \
#--debug \
#--dataset_name qed \
#--load_checkpoint_path ../resource/checkpoint/translation_qed/epoch\=56-step\=77690.ckpt \
#--checkpoint_dir ../resource/checkpoint/translation_qed

#python train_translation.py \
#--max_epochs 0 \
#--debug \
#--dataset_name drd2 \
#--load_checkpoint_path ../resource/checkpoint/translation_drd2/epoch\=72-step\=39054.ckpt \
#--checkpoint_dir ../resource/checkpoint/translation_drd2