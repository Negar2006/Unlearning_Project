import os
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel

BASE_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
UNLEARNED_TRANSFORMER_PATH = "outputs/sd3_duo_unlearned/transformer"
PROMPT_FILE = "data/test_prompts.txt"
OUTPUT_DIR = "outputs/unlearned"
BASE_SEED = 42
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 7.0
IMAGE_SIZE = 768

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(PROMPT_FILE):
    raise FileNotFoundError(f"File not found: {PROMPT_FILE}")

print("[1/3] Loading unlearned transformer weights...")
unlearned_transformer = SD3Transformer2DModel.from_pretrained(
    UNLEARNED_TRANSFORMER_PATH,
    torch_dtype=torch.bfloat16
)

print("[2/3] Loading SD3 base pipeline with unlearned transformer injected...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    BASE_MODEL_ID,
    transformer=unlearned_transformer,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
pipe.enable_model_cpu_offload()

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

print(f"[3/3] Generating unlearned images for {len(prompts)} prompts...")
for idx, prompt in enumerate(prompts):
    generator = torch.Generator(device="cpu").manual_seed(BASE_SEED + idx)
    print(f"Generating ({idx + 1}/{len(prompts)}) [Seed: {BASE_SEED + idx}]: '{prompt[:40]}...'")

    image = pipe(
        prompt=prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        generator=generator,
    ).images[0]

    image.save(os.path.join(OUTPUT_DIR, f"unlearned_{idx:03d}.png"))

print(f"\nFinished! All unlearned images are saved in '{OUTPUT_DIR}'.")