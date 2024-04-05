#!/bin/bash

DATA=$1 #first arg after running file in terminal: datapath_name where .img lists are saved, e.g. ./get_features_test.sh raw/YLFW_bench

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
python predict_magface.py --inf_list ../data/${DATA}/img.list \
                    --feat_list ../data/${DATA}/feat.list \
                    --batch_size 1 \
                    --resume /work3/s174139/Master_Thesis/MagFace-main/inference/magface_iresnet18_casia_dp.pth


#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python predict_magface.py --inf_list ../data/raw/RLFW_mini/img.list --feat_list ../data/raw/RLFW_mini/feat.list --batch_size 1 --resume /work3/s174139/Master_Thesis/MagFace-main/inference/magface_iresnet18_casia_dp.pth