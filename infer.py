import os
import torch

from pathlib import Path
from uuid import uuid4
from diffusers import StableDiffusionKDiffusionPipeline
from peft import PeftModel, LoraConfig
from diffusers.models import AutoencoderKL

# https://huggingface.co/docs/peft/task_guides/dreambooth_lora
def get_lora_sd_pipeline(
    ckpt_dir,
    base_model_name_or_path=None,
    dtype=torch.float32,
    device="cuda",
    adapter_name="default"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    # VAE
    vae = AutoencoderKL.from_pretrained("ShoukanLabs/OpenNiji", subfolder="vae").to(device)
    # Stable Diffusion
    pipe = StableDiffusionKDiffusionPipeline.from_pretrained(base_model_name_or_path, vae=vae, torch_dtype=dtype).to(device)
    # DPM++ 2M Karras
    # https://huggingface.co/stabilityai/stable-diffusion-2-1/discussions/23#645394effef4d71cdd6f7465
    pipe.set_scheduler('sample_dpmpp_2m')
    # EasyNegative embedding
    pipe.load_textual_inversion("/workspace/.cache/huggingface/EasyNegative.pt", weight_name="EasyNegative.pt", token="EasyNegative")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe

def main():
    model_path = Path("/workspace/model")
    if not os.path.exists(model_path):
        raise RuntimeError("`model` directory does not exist")
    pipe = get_lora_sd_pipeline(model_path)

    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, False)

    prompt = "yoshino, 1girl, masterpiece"
    negative_prompt = "EasyNegative"
    name = uuid4()
    image_dir = f'/workspace/images/{name}'
    os.makedirs(image_dir)
    for i in range(0, 100):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=6, negative_prompt=negative_prompt).images[0]
        image.save(f"{image_dir}/{i:03d}.png")

if __name__ == '__main__':
    main()
