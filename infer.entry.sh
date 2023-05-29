#!/bin/bash --login

# cache
export TORCH_HOME=/workspace/.cache/torch
export HF_HOME=/workspace/.cache/huggingface

mkdir -p $TORCH_HOME
mkdir -p $HF_HOME

# download EasyNegative
if [ ! -f "$HF_HOME/EasyNegative.pt" ]; then
  wget -q --show-progress "https://huggingface.co/datasets/gsdf/EasyNegative/resolve/60067b257337df8d7879142d870944fe4c6ab20d/EasyNegative.pt" -O "$HF_HOME/EasyNegative.pt"
fi

. activate peft

python /workspace/infer.py
