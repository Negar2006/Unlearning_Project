import os
import json
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

CLEAN_CSV = "Six-CD/Datasets/Dual-Version/Nudity/clean.csv"
IMAGES_DIR = "outputs/eval_baseline_utility"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP model: {CLIP_MODEL_NAME}...")
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

df = pd.read_csv(CLEAN_CSV)
prompts = df["prompt"].tolist() 

clip_scores = []

print("Calculating CLIP alignment scores...")
for idx, prompt in enumerate(tqdm(prompts)):
    img_path = os.path.join(IMAGES_DIR, f"clean_{idx:04d}.png")
    if not os.path.exists(img_path):
        continue
        
    image = Image.open(img_path).convert("RGB")
    
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Cosine similarity between image and text features
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        # Normalization
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        score = (image_embeds * text_embeds).sum().item()
        clip_scores.append(score)

mean_clip = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
print(f"\n[Result] Mean CLIP Alignment Score: {mean_clip:.4f}")