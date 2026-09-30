import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
PAIRS_DIR = "data/duo_pairs"
OUTPUT_LORA_DIR = "outputs/duo_lora"
os.makedirs(OUTPUT_LORA_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LEARNING_RATE = 1e-4
STEPS = 120

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class DUOPairDataset(Dataset):
    def __init__(self, base_dir):
        self.unsafe_dir = os.path.join(base_dir, "unsafe")
        self.safe_dir = os.path.join(base_dir, "safe")
        self.files = sorted(os.listdir(self.unsafe_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img_neg = transform(Image.open(os.path.join(self.unsafe_dir, f)).convert("RGB"))
        img_pos = transform(Image.open(os.path.join(self.safe_dir, f)).convert("RGB"))
        return img_pos, img_neg

dataset = DUOPairDataset(PAIRS_DIR)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

print("[1/3] Loading SD3 base pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    local_files_only=True
).to(DEVICE)
pipe.vae.eval()

print("Pre-computing explicit concept embeddings (T5 + CLIP)...")
explicit_nude_prompt = "a photo of a nude person, naked body, unclothed figure, bare skin"
with torch.no_grad():
    (
        nude_prompt_embeds,
        _,
        nude_pooled_embeds,
        _,
    ) = pipe.encode_prompt(
        prompt=explicit_nude_prompt,
        prompt_2=explicit_nude_prompt,
        prompt_3=explicit_nude_prompt,
        device=DEVICE
    )
    nude_prompt_embeds = nude_prompt_embeds.to(dtype=DTYPE)
    nude_pooled_embeds = nude_pooled_embeds.to(dtype=DTYPE)

del pipe.text_encoder
del pipe.text_encoder_2
del pipe.text_encoder_3
torch.cuda.empty_cache()

pipe.transformer.enable_gradient_checkpointing()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none"
)

trainable_transformer = get_peft_model(pipe.transformer, lora_config)
trainable_transformer.print_trainable_parameters()
trainable_transformer.train()

optimizer = torch.optim.AdamW(trainable_transformer.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

print("[2/3] Training Concept Redirection (Nude Concept -> Clothed Latents)...")
step = 0
data_iter = iter(dataloader)

while step < STEPS:
    try:
        x_pos, x_neg = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        x_pos, x_neg = next(data_iter)

    x_pos = x_pos.to(DEVICE, dtype=DTYPE)
    x_neg = x_neg.to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        lat_pos = (pipe.vae.encode(x_pos).latent_dist.sample()) * pipe.vae.config.scaling_factor
        lat_neg = (pipe.vae.encode(x_neg).latent_dist.sample()) * pipe.vae.config.scaling_factor

    t = torch.randint(250, 750, (1,), device=DEVICE).long()
    noise = torch.randn_like(lat_pos)

    sigma = (t.float() / 1000.0).to(dtype=DTYPE)
    x_neg_t = (1.0 - sigma) * lat_neg + sigma * noise
    
    target_v_to_safe = (noise - lat_pos).to(dtype=DTYPE)

    pred = trainable_transformer(
        hidden_states=x_neg_t,
        timestep=t,
        encoder_hidden_states=nude_prompt_embeds,
        pooled_projections=nude_pooled_embeds,
        return_dict=False
    )[0]

    loss = F.mse_loss(pred, target_v_to_safe)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_transformer.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 15 == 0:
        print(f"Step [{step:03d}/{STEPS}] | Redirection Loss: {loss.item():.4f}")

    step += 1

trainable_transformer.save_pretrained(OUTPUT_LORA_DIR)
print(f"[3/3] Training finished. LoRA saved in '{OUTPUT_LORA_DIR}'.")