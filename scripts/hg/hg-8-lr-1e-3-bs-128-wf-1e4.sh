#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID hg-8-lr-1e-3-bs-128-wf-1e4 \
  -nEpochs 50 \
  -batchSize 128 \
  -weightFocal 10000 \
  -LR 1e-3 \
  -netType hg-8
