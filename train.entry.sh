#!/bin/bash --login

# cache
export TORCH_HOME=/workspace/.cache/torch
export HF_HOME=/workspace/.cache/huggingface

INSTANCE_DIR="/workspace/instances"
OUTPUT_DIR="/workspace/model"

mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"
mkdir -p "$INSTANCE_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p /workspace/images

MODEL_NAME="stabilityai/stable-diffusion-xl-base-0.9"
INSTANCE_PROMPT="yoshino"

# login
if [ ! -z "$HUGGINGFACE_TOKEN" ]; then
  huggingface-cli login --token $HUGGINGFACE_TOKEN
fi

# Basic Usage
# https://github.com/huggingface/diffusers/blob/089bf7777998f53c0f0af4ae887b127dfec3ef50/examples/dreambooth/README_sdxl.md
# Parameter Tweaks
# https://github.com/huggingface/diffusers/blob/089bf7777998f53c0f0af4ae887b127dfec3ef50/examples/dreambooth/train_dreambooth_lora_sdxl.py
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="$INSTANCE_PROMPT" \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --use_8bit_adam \
  --seed="0"
