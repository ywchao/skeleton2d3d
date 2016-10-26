#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID hg-256-res-64-fts3-s3fix-hg-proj-w1e-6 \
  -dataset penn-crop \
  -data ./data/penn-crop \
  -nEpochs 10 \
  -batchSize 6 \
  -weightProj 1e-6 \
  -LR 2.5e-4 \
  -netType hg-256-res-64 \
  -hg \
  -hgs3Model exp/h36m/hg-256-res-64-h36m-hgfix-w1/model_best.t7 \
  -s3Fix \
  -evalOut hg
