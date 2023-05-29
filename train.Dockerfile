
FROM huggingface/peft-gpu

RUN apt update
RUN apt install -y unzip git wget

# user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# create user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
# set user
USER $USERNAME

WORKDIR /app/
RUN git clone https://github.com/huggingface/peft
WORKDIR /app/peft
RUN git reset --hard 3714aa2fff158fdfa637b2b65952580801d890b2

WORKDIR /app/peft/examples/lora_dreambooth
RUN /bin/bash -c ". activate peft && pip install -r requirements.txt"

# StableDiffusionKDiffusionPipeline sample_dpmpp_2m
RUN /bin/bash -c ". activate peft && pip install k-diffusion"

COPY --chown=$USERNAME train.entry.sh .
RUN chmod +x train.entry.sh

ENTRYPOINT ["/bin/bash", "-c", "./train.entry.sh"]
