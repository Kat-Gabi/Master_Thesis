#!/usr/bin/env bash
# If you don't have acess to run, run: tr -d '\r' <run_fine_tuner.sh > run_fine_tuner_new.sh
# To create new executable file
# Then run chmod +x run_fine_tuner_new.sh
# Consider to replace woth the same name again. 

# Submit this using bsub < jobscript.sh

echo "Starting fine tuner script..."


#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1


la=10
ua=110
lm=0.45
um=0.8
lg=35

# settings
MODEL_ARC=iresnet18
OUTPUT=./test_finetuning/
PRETRAINED_MODEL=../models/magface_iresnet18_casia_dp.pth
TRAIN_LIST= ../../data/data_full/GanDiffFace_processed_cluster_magface/fine_tune_train_list_magface_all_GanDiffFace.list #../../data/data_full/HDA_processed_cluster_magface/fine_tune_train_list_magface_all_HDA.list


# Check if pretrained model file exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Error: pretrained model file '$PRETRAINED_MODEL' not found."
    exit 1
fi

# Check if train_list file exists
if [ ! -f "$TRAIN_LIST" ]; then
    echo "Error: train_list file '$TRAIN_LIST' not found."
    exit 1
fi


#85 before!!
# Create output directory if it doesn't exist
mkdir -p "${OUTPUT}/vis/"


python -u ../run/fine_tuner.py \
    --arch ${MODEL_ARC} \
    --train_list ${TRAIN_LIST} \
    --pretrained ${PRETRAINED_MODEL} \
    --cpu_mode 0 \
    --workers 8 \
    --epochs 30 \
    --start-epoch 0 \
    --batch-size 256 \
    --embedding-size 512 \
    --last-fc-size 1652 \
    --arc-scale 64 \
    --learning-rate 0.01 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --lr-drop-epoch 12 18 26 \
    --lr-drop-ratio 0.1 \
    --print-freq 100 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 1 \
    --l_a ${la} \
    --u_a ${ua} \
    --l_margin ${lm} \
    --u_margin ${um} \
    --lambda_g ${lg} \
    --vis_mag 1    2>&1 | tee ${OUTPUT}/output.log   