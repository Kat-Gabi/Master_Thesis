#!/bin/bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
python gen_feat.py --inf_list toy_imgs_2/img.list --feat_list toy_imgs_2/feat.list --batch_size 1 --resume magface_iresnet18_casia_dp.pth

#Taken from eval.sh

# CKPT=$1
# FEAT_SUFFIX=$2
# *NL=$3 # argument 3. Her tast 0 for at få resnet18 - index 0 in iresnet.py list. Resnet18 automatically also trains on Casia-Webface (see github README)

# *ARCH=iresnet${NL}
# FEAT_PATH=./features/magface_${ARCH}/ #der hvor features bliver gemt
# mkdir -p ${FEAT_PATH}

# python3 ../../inference/gen_feat.py --arch ${ARCH} \
#                     --feat_list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \ 
#                     --batch_size 256 \
#                     --resume ${CKPT}