#!/usr/bin/env bash
echo "Starting fine tuner script..."


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

la=10
ua=110
lm=0.45
um=0.8
lg=35

# settings
MODEL_ARC=iresnet18
OUTPUT=./test/
PRETRAINED_MODEL=../../MagFace-main/inference/magface_iresnet18_casia_dp.pth
TRAIN_LIST=../../data/fine_tune_train_list.list



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



# Create output directory if it doesn't exist
mkdir -p "${OUTPUT}/vis/"


python -u fine_tuner.py \
    --arch ${MODEL_ARC} \
    --train_list ../../data/fine_tune_train_list.list \
    --pretrained ${PRETRAINED_MODEL} \
    --cpu_mode 1 \
    --workers 8 \
    --epochs 2 \
    --start-epoch 0 \
    --batch-size 6 \
    --embedding-size 6 \
    --last-fc-size 2 \
    --arc-scale 64 \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --lr-drop-epoch 2 \
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