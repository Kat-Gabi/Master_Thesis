#!/bin/bash

DATA=$1 #first arg after running file in terminal: ./get_features_test.sh + adults or children 

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
python predict_magface.py --inf_list ../data/raw_full/img_${DATA}.list \
                    --feat_list ../data/feat_${DATA}.list \
                    --batch_size 128 \
                    --resume /work3/s174139/Master_Thesis/MagFace-main/inference/magface_iresnet18_casia_dp.pth


#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python predict_magface.py --inf_list ../data/raw/RLFW_mini/img.list --feat_list ../data/raw/RLFW_mini/feat.list --batch_size 1 --resume /work3/s174139/Master_Thesis/MagFace-main/inference/magface_iresnet18_casia_dp.pth