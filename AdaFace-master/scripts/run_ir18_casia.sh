
python ../main_master.py \
    --data_root ../../data/data_full/ \
    --train_data_path HDA_processed_AdaFace/ \
    --prefix /Users/gabriellakierulff/anaconda3/envs/best_master \
    --use_wandb \
    --gpus 0 \
    --use_16bit \
    --arch ir_18 \
    --batch_size 512 \
    --num_workers 8 \
    --epochs 2 \
    --lr_milestones 12,20,24 \
    --lr 0.1 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2 \
    --custom_num_class 25 \
    --start_from_model_statedict ../pretrained/adaface_ir18_casia.ckpt


