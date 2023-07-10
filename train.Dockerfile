
# https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/docker/diffusers-pytorch-cuda/Dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   libsndfile1-dev \
                   libgl1 \
                   python3.8 \
                   python3-pip \
                   python3.8-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        torch \
        torchvision \
        torchaudio \
        invisible_watermark && \
    python3 -m pip install --no-cache-dir \
        accelerate \
        datasets \
        hf-doc-builder \
        huggingface-hub \
        Jinja2 \
        librosa \
        numpy \
        scipy \
        tensorboard \
        transformers \
        omegaconf \
        pytorch-lightning \
        xformers

CMD ["/bin/bash"]
# ------

RUN apt update
RUN apt install -y unzip git wget libgl1

# user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# create user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
# set user
USER $USERNAME

# app
WORKDIR /app/
RUN git clone https://github.com/huggingface/diffusers
WORKDIR /app/diffusers
# https://github.com/huggingface/diffusers/commits/dreambooth/sd-xl-3
RUN git reset --hard 82b4cb51d69ee644f1e30c2d982d88f0702bd620

# https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
USER root
# pip install for the repository 
WORKDIR /app/diffusers
RUN python3 -m pip install -e .
# pip install for the example
WORKDIR /app/diffusers/examples/dreambooth
RUN python3 -m pip install -r requirements.txt
# To use 8-bit Adam
RUN python3 -m pip install bitsandbytes

# app
USER $USERNAME

COPY --chown=$USERNAME train.entry.sh .
RUN chmod +x train.entry.sh

ENTRYPOINT ["/bin/bash", "-c", "./train.entry.sh"]
