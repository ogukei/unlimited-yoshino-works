#!/bin/bash --login

# cache
export TORCH_HOME=/workspace/.cache/torch
export HF_HOME=/workspace/.cache/huggingface

mkdir -p $TORCH_HOME
mkdir -p $HF_HOME
mkdir -p /workspace/instances
mkdir -p /workspace/classes
mkdir -p /workspace/model
mkdir -p /workspace/images

MODEL_NAME_OR_PATH="Korakoe/OpenNiji"
INSTANCE_PROMPT="yoshino"
CLASS_PROMPT="1girl, brown hair, brown eyes, long hair, very long hair, blunt bangs, white background, simple background"

. activate peft

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="$MODEL_NAME_OR_PATH" \
  --instance_data_dir=/workspace/instances \
  --class_data_dir=/workspace/classes \
  --output_dir=/workspace/model \
  --train_text_encoder \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="$INSTANCE_PROMPT" \
  --class_prompt="$CLASS_PROMPT" \
  --resolution=512 \
  --train_batch_size=1 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_num_cycles=4 \
  --lr_warmup_steps=500 \
  --num_class_images=200 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 27 \
  --lora_text_encoder_r 16 \
  --lora_text_encoder_alpha 17 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --mixed_precision=fp16 \
  --max_train_steps=600
