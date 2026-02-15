#!/bin/bash

python /home/kotsios/dsit/thesis/thesis_project/src/scripts/train/train_model.py \
  --backbone_model "jinaai/jina-embeddings-v2-small-en" \
  --backbone_model_output_dim 512 \
  --target_dim 32 \
  --epochs 10 \
  --learning_rate 0.01 \
  --positional_loss_factor 1.0 \
  --train_batch_size 20000 \
  --val_batch_size 20000 \
  --lr_scheduler_type "linear" \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --weight_exponent 1
  # --custom_suffix "" \
  # --resume_from_checkpoint 
