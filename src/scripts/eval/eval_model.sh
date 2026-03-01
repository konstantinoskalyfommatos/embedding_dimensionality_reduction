#!/bin/bash

# --backbone_model Alibaba-NLP/gte-multilingual-base \

python src/scripts/eval/eval_model.py \
    --backbone_model "jinaai/jina-embeddings-v2-small-en" \
    --backbone_model_output_dim 512 \
    --checkpoint 4800 \
    --target_dim 32 \
    --positional_loss_factor 1.0 \
    --train_batch_size 20000 \
    --sts_batch_size 2048 \
    --retrieval_batch_size 6 \
    --classification_batch_size 20 \
    --clustering_batch_size 16 \
    --spearman_test_batch_size 5000 \
    --custom_suffix "" \
    # --intrinsic \
    # --skip_sts \
    # --skip_classification \
    # --skip_retrieval \
    # --skip_clustering \
    # --overwrite_cache \
    # --fast_mode