import os
import json
import argparse
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel

NUM_STEPS = 28
GUIDANCE_SCALE = 7.0
RESOLUTION = 768
EVAL_FILE = "data/evaluation_set.json"
BASE_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["baseline", "unlearned"], required=True)
args = parser.parse_args()

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_items = json.load(f)

output_dir = f"outputs/eval_{args.mode}"
os.makedirs(output_dir, exist_ok=True)

print(f"Loading pipeline for mode: '{args.mode}'...")

if args.mode == "baseline":
    pipe = StableDiffusion3Pipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
else:
    unlearned_transformer = SD3Transformer2DModel.from_pretrained(
        "outputs/sd3_duo_unlearned/transformer",
        torch_dtype=torch.bfloat16
    )
    pipe = StableDiffusion3Pipeline.from_pretrained(
        BASE_MODEL_ID,
        transformer=unlearned_transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

pipe.enable_model_cpu_offload()

print(f"Starting generation for {len(eval_items)} benchmark prompts...")

for idx, item in enumerate(eval_items):
    case_num = item["case_number"]
    seed = item["evaluation_seed"]
    prompt = item["prompt"]
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    print(f"[{idx+1}/{len(eval_items)}] Case {case_num} | Seed: {seed} | '{prompt[:45]}...'")
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=RESOLUTION,
        width=RESOLUTION,
        generator=generator
    ).images[0]
    
    save_path = os.path.join(output_dir, f"case_{case_num:04d}.png")
    image.save(save_path)

print(f"All images for {args.mode} generated successfully in '{output_dir}'.")