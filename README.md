# Unlimited Yoshino Works

Generates Yorita Yoshino images unlimitedly.

Actual output |
:-:|
![yoshino](https://gist.githubusercontent.com/ogukei/07c3262baee88c3214e4d272289ef3e2/raw/2fc36b262f4bc17e40832d29d19635955deb4b0c/yoshino.png) |

https://huggingface.co/docs/peft/task_guides/dreambooth_lora

## Requirements

* Linux (Ubuntu 22.04 LTS) or Windows
* NVIDIA Container Toolkit
    * https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian

## Usage

1. Place Yorita Yoshino images in the `./instances` directory
    * The recommended number of images is around 10 to 20
    * Use images with a white background, not a transparent background
2. You need to specify your HuggingFace token to the environment variable `HUGGINGFACE_TOKEN` in order to download SDXL 0.9 from the hub at the moment
    * via https://huggingface.co/settings/tokens
3. Run the following command to train our model

```
docker compose run --build -e HUGGINGFACE_TOKEN=<YOUR_TOKEN> train
```

3. Make sure the file `./model/pytorch_lora_weights.bin` is generated
4. Run the following command to generate images

```
docker compose run --build -e HUGGINGFACE_TOKEN=<YOUR_TOKEN> infer
```

5. Make sure `./images/<uuid>/000.png` and so on are generated

## Tweak LoRA

1. Put the character images in the `./instances` directory
1. Enter the character name into the INSTANCE_PROMPT variable in the `./train.entry.sh` script.
1. Enter the common characteristics among your instance images into the CLASS_PROMPT variable.
    * e.g. `brown hair, brown eyes, long hair, white background`
1. Edit the variable `prompt` to specify what you want to generate in the script `./infer.py`
