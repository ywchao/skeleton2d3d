#!/bin/bash

unzip smplify_code.zip
unzip SMPL_python_v.1.0.0.zip
tar -zxvf lsp_results.tar.gz -C smplify_public

cd smplify_public
mkdir images
ln -s /z/ywchao/datasets/lsp_dataset/images images/lsp
ln -s /z/ywchao/codes/image-play/skeleton2d3d/smplify/smpl/models/*.pkl code/models
cd ..
