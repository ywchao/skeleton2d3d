#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID hg-256-res-64-h36m-w0 \
  -nEpochs 100 \
  -batchSize 16 \
  -weightHMap 0 \
  -weightProj 0 \
  -netType hg-256-res-64 \
  -penn \
  -hg \
  -hgModel ../pose-hg-train/exp/penn_action_cropped/hg-256-ft/best_model.t7 \
  -s3Model ./exp/h36m/res-64-t2/model_best.t7
