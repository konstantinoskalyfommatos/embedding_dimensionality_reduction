#!/bin/bash

# --backbone_model Alibaba-NLP/gte-multilingual-base

python src/scripts/eval/eval_random_selection.py \
    --backbone_model "jinaai/jina-embeddings-v2-small-en" \
    --backbone_model_output_dim 512 \
    --target_dim 32 \
    --sts_batch_size 2048 \
    --classification_batch_size 6 \
    --retrieval_batch_size 20 \
    --clustering_batch_size 16 \
    --weight_exponent 0 \
    --positional_or_angular "angular" \
    --eval_intrinsic
    # --skip_sts \
    # --skip_classification \
    # --skip_retrieval \
    # --skip_clustering \
    # --fast_mode \
