#!/usr/bin/env bash

PROJ_ROOT=/home/jhair/Research/DOING/Neuro_Inspired_SSCL__dev__/isolated_experiments/SSL_on_episodes/offline_training

export PYTHONPATH=${PROJ_ROOT}
source activate py39gpu
cd ${PROJ_ROOT}

VIEWS=2

python -u main.py \
    --data_path  /data/datasets/ImageNet-100 \
    --model_name resnet18 \
    --pretrained_model None \
    --num_pseudoclasses 100 \
    --num_views ${VIEWS} \
    --epochs 50 \
    --stop_epoch 10 > logs_main/log_scratch_${VIEWS}views.log