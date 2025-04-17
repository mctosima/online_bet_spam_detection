#!/usr/bin/env bash

# Define ablation parameters and their values
declare -a ablation_list=(
  "num_layers:1,2,3"
  "embed_dim:100,128"
  "ff_dim:256,512"
  "num_heads:2,4"
  "dropout:0.1,0.2,0.3"
  "pooling_strategy:cls,mean"
  "lr:1e-5,5e-5,1e-4"
  "optimizer:adam,adamw,sgd"
  "batch_size:16,32,64"
  "scheduler:step_lr,reduce_lr,cosine"
  "class_weights:true,false"
  "augmentation:none,random,cascaded"
)

# Base command
BASE_CMD="python ycj_train.py --model_type lightweight --use_wandb"

for entry in "${ablation_list[@]}"; do
  IFS=":" read -r param values <<< "$entry"
  echo -e "\n===== Ablation on $param [$values] ====="
  $BASE_CMD \
    --ablation \
    --ablation_param "$param" \
    --ablation_values "$values"
done
