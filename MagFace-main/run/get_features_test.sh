#!/bin/bash

# Worked BEFORE /with cropping !!
# # Set the desired working directory
# cd /work3/s174139/Master_Thesis/MagFace-main/

# DATA=$1 #first arg after running file in terminal: ./get_features_test.sh + adults or childrenR. sometimes run dos2unix get_features_test.sh to get access 

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
# python run/predict_magface.py --inf_list ../data/data_full/RFW/${DATA}.list \
#                     --feat_list ../data/data_full/feature_vectors/magface_feature_vectors/feature_vectors_from_${DATA}.list \
#                     --batch_size 128 \
#                     --resume /work3/s174139/Master_Thesis/MagFace-main/models/magface_iresnet18_casia_dp.pth

# #PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python predict_magface.py --inf_list ../data/raw/RLFW_mini/img.list --feat_list ../data/raw/RLFW_mini/feat.list --batch_size 1 --resume /work3/s174139/Master_Thesis/MagFace-main/inference/magface_iresnet18_casia_dp.pth


## FINETUNING
# # Set the desired working directory
# cd /work3/s174139/Master_Thesis/MagFace-main/

# DATA=$1 #first arg after running file in terminal: ./get_features_test.sh + adults or childrenR. sometimes run dos2unix get_features_test.sh to get access 

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
# python run/predict_magface_original.py --inf_list ../data/data_full/feature_vectors/magface_image_lists/${DATA}.list \
#                     --feat_list ../data/data_full/feature_vectors/magface_feature_vectors/feature_vectors_from_${DATA}_finetuning_ex_2_1.list \
#                     --batch_size 128 \
#                     --resume /work3/s174139/Master_Thesis/MagFace-main/Master_thesis_data_prep/test_finetuning/00030_USE_FINAL.pth

# #PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python predict_magface.py --inf_list ../data/raw/RLFW_mini/img.list --feat_list ../data/raw/RLFW_mini/feat.list --batch_size 1 --resume /work3/s174139/Master_Thesis/MagFace-main/inference/magface_iresnet18_casia_dp.pth

cd /work3/s174139/Master_Thesis/MagFace-main/

DATA=$1 #first arg after running file in terminal: ./get_features_test.sh + adults or childrenR. sometimes run dos2unix get_features_test.sh to get access 

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
python run/predict_magface_original.py --inf_list ../data/data_full/feature_vectors/magface_image_lists/${DATA}.list \
                    --feat_list ../data/data_full/feature_vectors/magface_feature_vectors/feature_vectors_from_${DATA}_baseline_ex_1_1.list \
                    --batch_size 128 \
                    --resume /work3/s174139/Master_Thesis/MagFace-main/models/magface_iresnet18_casia_dp.pth

#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python predict_magface.py --inf_list ../data/raw/RLFW_mini/img.list --feat_list ../data/raw/RLFW_mini/feat.list --batch_size 1 --resume /work3/s174139/Master_Thesis/MagFace-main/inference/magface_iresnet18_casia_dp.pth
