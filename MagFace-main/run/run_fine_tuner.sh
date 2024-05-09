#!/usr/bin/env bash
# If you don't have acess to run, run: tr -d '\r' <run_fine_tuner.sh > run_fine_tuner_new.sh
# To create new executable file
# Then run chmod +x run_fine_tuner_new.sh
# Consider to replace woth the same name again. 
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
PRETRAINED_MODEL=../models/magface_iresnet18_casia_dp.pth
TRAIN_LIST=../../data/data_full/HDA_aligned_resized/fine_tune_train_list.list


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
    --train_list ${TRAIN_LIST} \
    --pretrained ${PRETRAINED_MODEL} \
    --cpu_mode 1 \
    --workers 1 \
    --epochs 5 \
    --start-epoch 0 \
    --batch-size 256 \
    --embedding-size 512 \
    --last-fc-size 2 \
    --arc-scale 64 \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --lr-drop-epoch 2 \
    --lr-drop-ratio 0.1 \
    --print-freq 10 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 1 \
    --l_a ${la} \
    --u_a ${ua} \
    --l_margin ${lm} \
    --u_margin ${um} \
    --lambda_g ${lg} \
    --vis_mag 1    2>&1 | tee ${OUTPUT}/output.log   