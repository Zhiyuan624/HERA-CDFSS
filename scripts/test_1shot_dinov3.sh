#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main_hera.py \
  --test_datapath ./data/deepglobe \
  --backbone DINOv3 \
  --benchmark deepglobe \
  --fold 0 \
  --nshot 1 \
  --refine always \
  --fusion on \
  --feat_id 12 13 14 15 16 17 18 19 20 21 22 23 \
  --attn_strategy dual_attn_gauss \
  --logdir ./logs/deepglobe \
  --logfile Dinov3_deepglobe_shot1.txt &

CUDA_VISIBLE_DEVICES=1 python main_hera.py \
  --test_datapath ./data/ISIC \
  --backbone DINOv3 \
  --benchmark isic \
  --fold 0 \
  --nshot 1 \
  --refine always \
  --fusion on \
  --feat_id 12 13 14 15 16 17 18 19 20 21 22 23 \
  --attn_strategy dual_attn_gauss \
  --logdir ./logs/isic \
  --logfile Dinov3_isic_shot1.txt &

CUDA_VISIBLE_DEVICES=2 python main_hera.py \
  --test_datapath ./data/chest \
  --backbone DINOv3 \
  --benchmark lung \
  --fold 0 \
  --nshot 1 \
  --refine always \
  --fusion on \
  --feat_id 12 13 14 15 16 17 18 19 20 21 22 23 \
  --attn_strategy dual_attn_gauss \
  --logdir ./logs/chest \
  --logfile Dinov3_lung_shot1.txt &

CUDA_VISIBLE_DEVICES=3 python main_hera.py \
  --test_datapath ./data/fss \
  --backbone DINOv3 \
  --benchmark fss \
  --fold 0 \
  --nshot 1 \
  --refine always \
  --fusion on \
  --feat_id 12 13 14 15 16 17 18 19 20 21 22 23 \
  --attn_strategy dual_attn_gauss \
  --logdir ./logs/images \
  --logfile Dinov3_fss_shot1.txt &