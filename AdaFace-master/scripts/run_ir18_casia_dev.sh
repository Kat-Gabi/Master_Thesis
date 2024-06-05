
#WANDB_DIR=/work3/s174139/best_master_remote/bin/activate_05-15_0/wandb/
#    --use_wandb \

# Locate: /Users/gabriellakierulff/anaconda3/envs/best_master \
# Submit this using bsub < jobscript.sh

python ../main_master.py \
    --data_root ../../data/data_full/ \
    --train_data_path HDA_processed_AdaFace_development/ \
    --prefix ir18_casia_adaface \
    --gpus 1 \
    --use_16bit \
    --arch ir_18 \
    --batch_size 256 \
    --num_workers 4 \
    --epochs 3 \
    --lr_milestones 1,2 \
    --lr 0.01 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2 \
    --save_all_models \
    --custom_num_class 91 \
    --start_from_model_statedict ../pretrained/adaface_ir18_casia.ckpt


