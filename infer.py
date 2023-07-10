
from diffusers import StableDiffusionXLPipeline
from uuid import uuid4
import torch
import os

def main():
    name = uuid4()
    image_dir = f'/workspace/images/{name}'

    lora_weights_dir = "/workspace/model"
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16)
    pipe.load_lora_weights(lora_weights_dir)
    pipe.to("cuda")

    prompt = "yoshino, 1girl, brown hair, brown eyes, long hair, very long hair, blunt bangs, white background"

    os.makedirs(image_dir)
    for i in range(0, 100):
        image = pipe(prompt, num_inference_steps=30, guidance_scale=9.0).images[0]
        image.save(f"{image_dir}/{i:03d}.png")

if __name__ == '__main__':
    main()
