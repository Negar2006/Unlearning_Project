import os
import torch
from diffusers import SD3Transformer2DModel
from peft import PeftModel

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
LORA_PATH = "outputs/duo_lora"
EXPORT_DIR = "outputs/sd3_duo_unlearned/transformer"

os.makedirs(EXPORT_DIR, exist_ok=True)

print("[1/3] Loading SD3 base Transformer...")
transformer = SD3Transformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    local_files_only=True
)

print("[2/3] Merging DUO LoRA weights into MMDiT...")
transformer = PeftModel.from_pretrained(transformer, LORA_PATH)
transformer = transformer.merge_and_unload()

print(f"[3/3] Saving merged transformer to '{EXPORT_DIR}'...")
transformer.save_pretrained(EXPORT_DIR)

print("\nMerge complete! Only the unlearned transformer was saved successfully.")