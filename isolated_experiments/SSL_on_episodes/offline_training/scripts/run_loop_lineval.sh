#!/usr/bin/env bash

PROJ_ROOT=/home/jhair/Research/DOING/Neuro_Inspired_SSCL__dev__/isolated_experiments/SSL_on_episodes/offline_training

export PYTHONPATH=${PROJ_ROOT}
source activate py39gpu
cd ${PROJ_ROOT}

GPU=0
RUN_TYPE=only_random_crops

for NUM_VIEWS in 2 4 8 12; do
  for EPOCH_CHECK in init epoch0 epoch1 epoch2 epoch3 epoch4 epoch5 epoch6 epoch7 epoch8 epoch9; do
      PRETRAINED_FOLDER=output/${RUN_TYPE}/fromScratch_50epochs_${NUM_VIEWS}views_0.25lr_128bs
      PRETRAINED_MODEL=${PRETRAINED_FOLDER}/encoder_${EPOCH_CHECK}.pth

      echo "VIEWS: ${NUM_VIEWS}; Pretrained model: ${PRETRAINED_MODEL}"
      SAVE_DIR=${PRETRAINED_FOLDER}/lineval/lineval_encoder_${EPOCH_CHECK}/
      LOG_FILE=lineval_${NUM_VIEWS}views_encoder_${EPOCH_CHECK}.log

      CUDA_VISIBLE_DEVICES=${GPU} python -u linear_eval.py \
        --pretrained_model ${PRETRAINED_MODEL} \
        --epochs 1 \
        --save_dir ${SAVE_DIR} > logs/${LOG_FILE}
  done
done