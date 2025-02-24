#!/bin/bash

# Set environment variables
export PYTHONPATH="${PWD}"
export CUDA_VISIBLE_DEVICES="0,1"
export ACCELERATE_USE_CPU="false"
export ACCELERATE_USE_CUDA="true"
export ACCELERATE_MIXED_PRECISION="fp16"
export ACCELERATE_CUDA_DEVICE="0"
export NUM_PROCESSES="2"
export WANDB_PROJECT="temporal-ae"
export WANDB_WATCH="gradients"
export WANDB_LOG_MODEL="true"

# Launch training script with accelerate
accelerate launch \
    examples/temporal_ae/train_temporal_ae.py \
    --pretrained_model_name_or_path=/data/42-julia-hpc-rz-cv/sig95vg/checkpoints/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/vae \
    --config_path=examples/temporal_ae/realbasicvsr.yaml \
    --resolution=256 \
    --enable_xformers_memory_efficient_attention \
    --validation_videos /home/sig95vg/codes/STAR/dataset/OpenVidHD/video/1XnfX9tTc8c_21_0to760.mp4 \
    --validation_steps=100 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=10 \
    --learning_rate=1e-5 \
    --output_dir=temporal-ae-finetuned \
    --mixed_precision=fp16 \
    --report_to=wandb \
    --checkpointing_steps=500 \
    --validation_steps=100 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=500 \
    --seed=42 \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --adam_weight_decay=1e-2 \
    --adam_epsilon=1e-08 \
    --kl_weight=0.000001 \
    --use_ema \
    --non_ema_revision=main
# /home/sig95vg/codes/STAR/dataset/OpenVidHD/video/3Vo5yQtm3PQ_14_0to165.mp4 \
# /home/sig95vg/codes/STAR/dataset/OpenVidHD/video/5CVSIa9606Y_2_1475to1620.mp4 \
# /home/sig95vg/codes/STAR/dataset/OpenVidHD/video/7dyMJwZQu6g_11_48to171.mp4 \
# /home/sig95vg/codes/STAR/dataset/OpenVidHD/video/9IggDJ3JxtU_4_16to152.mp4 \
# /home/sig95vg/codes/STAR/dataset/OpenVidHD/video/CVW6bNIo-CI_0_0to246.mp4 \
# /home/sig95vg/codes/STAR/dataset/OpenVidHD/video/FEbAwyrF1NM_0_0to112.mp4 \