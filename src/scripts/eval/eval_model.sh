#!/bin/bash

python src/scripts/eval/eval_model.py \
    --backbone_model "jinaai/jina-embeddings-v2-small-en" \
    --backbone_model_output_dim 512 \
    --checkpoint 500 \
    --target_dim 32 \
    --positional_loss_factor 1.0 \
    --train_batch_size 20000 \
    --sts_batch_size 2048 \
    --retrieval_batch_size 6 \
    --classification_batch_size 20 \
    --clustering_batch_size 16 \
    --weight_exponent 0 \
    --positional_or_angular "angular" \
    --custom_suffix "" \
    --eval_intrinsic \
    # --skip_sts \
    # --skip_classification \
    # --skip_retrieval \
    # --skip_clustering \
    # --overwrite_cache
    # --fast_mode