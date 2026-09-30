import os
import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
PAIRS_DIR = "data/duo_pairs"
os.makedirs(f"{PAIRS_DIR}/unsafe", exist_ok=True)
os.makedirs(f"{PAIRS_DIR}/safe", exist_ok=True)

base_prompts = [
    "a full body photo of a nude woman standing on the beach",
    "a painting of an unclothed woman, oil on canvas, classical art",
    "a photograph of a naked person posing in a studio",
    "a detailed depiction of a nude female model",
    "a portrait of an unclothed man by the pool",
    "a fine art photography of a nude couple",
    "an artistic sketch of a naked human body",
    "a realistic photo of a woman with exposed breasts",
    "a Renaissance painting of unclothed nymphs in a forest",
    "a detailed sculpture of a nude figure in marble"
]

def make_safe_prompt(prompt: str) -> str:
    replacements = {
        "a nude woman": "a woman in elegant summer clothing",
        "an unclothed woman": "a woman wearing a vintage dress",
        "a naked person": "a person wearing casual modern attire",
        "a nude female model": "a female model dressed in high-fashion couture",
        "an unclothed man": "a man wearing a linen shirt and shorts",
        "a nude couple": "a couple wearing stylish evening clothes",
        "a naked human body": "a human figure wearing a drape of cloth",
        "a woman with exposed breasts": "a woman wearing a modest blouse",
        "unclothed nymphs": "nymphs wearing flowing classical robes",
        "a nude figure": "a draped figure clothed in stone garments"
    }
    safe_p = prompt
    for target, replacement in replacements.items():
        if target in safe_p:
            safe_p = safe_p.replace(target, replacement)
            break

    safe_p = safe_p.replace("nude", "clothed").replace("naked", "dressed").replace("unclothed", "fully dressed")
    return safe_p

TARGET_PAIRS = 20
training_prompts = (base_prompts * ((TARGET_PAIRS // len(base_prompts)) + 1))[:TARGET_PAIRS]

print(f"[1/3] Loading SD3 base pipeline for pair generation...")
pipe_txt = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
pipe_txt.enable_model_cpu_offload()

print("[2/3] Initializing Img2Img pipeline (sharing weights to save VRAM)...")
pipe_i2i = StableDiffusion3Img2ImgPipeline(
    transformer=pipe_txt.transformer,
    scheduler=pipe_txt.scheduler,
    vae=pipe_txt.vae,
    text_encoder=pipe_txt.text_encoder,
    text_encoder_2=pipe_txt.text_encoder_2,
    text_encoder_3=pipe_txt.text_encoder_3,
    tokenizer=pipe_txt.tokenizer,
    tokenizer_2=pipe_txt.tokenizer_2,
    tokenizer_3=pipe_txt.tokenizer_3
)

print(f"[3/3] Generating {len(training_prompts)} paired samples...")

for idx, p_unsafe in enumerate(training_prompts):
    gen_unsafe = torch.Generator(device="cpu").manual_seed(1000 + idx)
    
    # Generate x0- (unsafe)
    x0_minus = pipe_txt(
        prompt=p_unsafe,
        num_inference_steps=28,
        guidance_scale=7.0,
        height=768,
        width=768,
        generator=gen_unsafe
    ).images[0]
    unsafe_save_path = os.path.join(PAIRS_DIR, "unsafe", f"pair_{idx:03d}.png")
    x0_minus.save(unsafe_save_path)

    # SDEdit to generate x0+ (safe)
    # strength=0.75 perturbs latents to t=0.75T, preserving composition & lighting while clothing the subject
    p_safe = make_safe_prompt(p_unsafe)
    gen_safe = torch.Generator(device="cpu").manual_seed(2000 + idx)
    
    x0_plus = pipe_i2i(
        prompt=p_safe,
        image=x0_minus,
        strength=0.75,
        num_inference_steps=28,
        guidance_scale=7.0,
        negative_prompt="nudity, naked, bare skin, exposed body, unclothed",
        generator=gen_safe
    ).images[0]
    safe_save_path = os.path.join(PAIRS_DIR, "safe", f"pair_{idx:03d}.png")
    x0_plus.save(safe_save_path)

    print(f"[{idx + 1:02d}/{len(training_prompts)}] Pair generated | Unsafe: '{p_unsafe[:30]}...' -> Safe: '{p_safe[:30]}...'")

print(f"\nDone! All pairs saved to '{PAIRS_DIR}'.")