"""
SimCLR Self-Supervised Pretraining - Google Colab Notebook

ルール適合: 本番入力画像のみ使用（ラベル不使用）

Usage:
1. Kaggle/Colabにコピー
2. GPU有効化
3. 実行

Phase 1: SimCLR事前学習 (100 epochs, ~2h on T4)
Phase 2: Fine-tune (30 epochs, ~1h on T4)
"""

# %% [markdown]
# # SimCLR Self-Supervised Pretraining
# 
# **ルール適合**: 自己教師あり事前学習（ラベル不使用）

# %% Install dependencies
!pip install -q segmentation-models-pytorch

# %% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import json
import cv2
import segmentation_models_pytorch as smp

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# %% Configuration
class Config:
    # Kaggle paths
    data_dir = Path("/kaggle/input/medical-ai-contest-7th-2025")
    output_dir = Path("/kaggle/working")
    
    image_size = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    
    # Phase 1: SimCLR
    simclr_epochs = 100
    simclr_batch_size = 32
    simclr_lr = 3e-4
    temperature = 0.5
    projection_dim = 128
    
    # Phase 2: Fine-tune
    finetune_epochs = 30
    finetune_batch_size = 8
    encoder_lr = 1e-5
    decoder_lr = 1e-4
    
    encoder_name = "efficientnet-b4"

config = Config()
config.output_dir.mkdir(exist_ok=True)

# %% SimCLR Augmentation
class SimCLRAugmentation:
    def __init__(self, size=512):
        self.size = size
    
    def __call__(self, img):
        # Random crop
        h, w = img.shape[:2]
        scale = np.random.uniform(0.8, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        top = np.random.randint(0, max(1, h - new_h + 1))
        left = np.random.randint(0, max(1, w - new_w + 1))
        img = img[top:top+new_h, left:left+new_w]
        img = cv2.resize(img, (self.size, self.size))
        
        # Flip
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
        if np.random.random() > 0.5:
            img = np.flipud(img).copy()
        
        # Rotate 90
        k = np.random.randint(0, 4)
        img = np.rot90(img, k).copy()
        
        # Blur
        if np.random.random() > 0.5:
            ksize = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        # Brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-0.1, 0.1)
        img = np.clip(alpha * img + beta, 0, 1)
        
        return img.astype(np.float32)

# %% SimCLR Dataset (NO LABELS!)
class SimCLRDataset(Dataset):
    def __init__(self, data_dir, size=512):
        self.data_dir = Path(data_dir)
        self.size = size
        self.augment = SimCLRAugmentation(size)
        
        # Collect ALL input images (train + test)
        self.paths = []
        
        train_csv = self.data_dir / "train.csv"
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            self.paths.extend([self.data_dir / p for p in df['input_path']])
        
        test_csv = self.data_dir / "test.csv"
        if test_csv.exists():
            df = pd.read_csv(test_csv)
            self.paths.extend([self.data_dir / p for p in df['input_path']])
        
        print(f"SimCLR: {len(self.paths)} images")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        img = img.resize((self.size, self.size))
        img = np.array(img, dtype=np.float32) / 255.0
        
        v1 = torch.from_numpy(self.augment(img)).unsqueeze(0)
        v2 = torch.from_numpy(self.augment(img)).unsqueeze(0)
        return v1, v2

# %% SimCLR Model
class SimCLRModel(nn.Module):
    def __init__(self, encoder_name, proj_dim=128):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=1, weights=None)
        enc_dim = self.encoder.out_channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(enc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )
    
    def forward(self, x):
        f = self.encoder(x)[-1]
        f = self.pool(f).flatten(1)
        z = self.proj(f)
        return F.normalize(z, dim=1)
    
    def get_encoder_weights(self):
        return self.encoder.state_dict()

# %% NT-Xent Loss
class NTXentLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp
    
    def forward(self, z1, z2):
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temp
        
        mask = torch.eye(2*N, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)
        
        labels = torch.cat([torch.arange(N) + N, torch.arange(N)]).to(z.device)
        return F.cross_entropy(sim, labels)

# %% Phase 1: SimCLR Training
print("="*60)
print("Phase 1: SimCLR Self-Supervised Pretraining")
print("="*60)

torch.manual_seed(config.seed)
np.random.seed(config.seed)

dataset = SimCLRDataset(config.data_dir, config.image_size)
loader = DataLoader(dataset, batch_size=config.simclr_batch_size, shuffle=True, 
                    num_workers=2, drop_last=True, pin_memory=True)

model = SimCLRModel(config.encoder_name, config.projection_dim).to(config.device)
criterion = NTXentLoss(config.temperature)
optimizer = optim.AdamW(model.parameters(), lr=config.simclr_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.simclr_epochs)

best_loss = float('inf')
for epoch in range(config.simclr_epochs):
    model.train()
    total_loss = 0
    
    for v1, v2 in tqdm(loader, desc=f"Epoch {epoch+1}"):
        v1, v2 = v1.to(config.device), v2.to(config.device)
        
        z1, z2 = model(v1), model(v2)
        loss = criterion(z1, z2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.get_encoder_weights(), config.output_dir / "simclr_encoder.pth")

print(f"\nSimCLR Training Complete! Best Loss: {best_loss:.4f}")
print(f"Encoder saved: {config.output_dir / 'simclr_encoder.pth'}")

# %% Phase 2: Fine-tune Dataset
class FinetuneDataset(Dataset):
    def __init__(self, csv_path, data_dir, size=512, indices=None):
        self.df = pd.read_csv(csv_path)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.size = size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        inp = Image.open(self.data_dir / row['input_path']).convert('L')
        inp = inp.resize((self.size, self.size))
        inp = np.array(inp, dtype=np.float32) / 255.0
        
        tgt = Image.open(self.data_dir / row['target_path']).convert('L')
        tgt = tgt.resize((self.size, self.size))
        tgt = np.array(tgt, dtype=np.float32) / 255.0
        
        return {
            "id": row["id"],
            "input": torch.from_numpy(inp).unsqueeze(0),
            "target": torch.from_numpy(tgt).unsqueeze(0)
        }

# %% Phase 2: Fine-tune
print("\n" + "="*60)
print("Phase 2: Fine-tune with SimCLR Encoder")
print("="*60)

# Split data
full_df = pd.read_csv(config.data_dir / "train.csv")
n_train = int(len(full_df) * 0.8)
train_ds = FinetuneDataset(config.data_dir / "train.csv", config.data_dir, config.image_size, list(range(n_train)))
val_ds = FinetuneDataset(config.data_dir / "train.csv", config.data_dir, config.image_size, list(range(n_train, len(full_df))))

train_loader = DataLoader(train_ds, batch_size=config.finetune_batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=config.finetune_batch_size, shuffle=False, num_workers=2)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# Create U-Net with SimCLR encoder
unet = smp.Unet(
    encoder_name=config.encoder_name,
    encoder_weights=None,
    in_channels=1,
    classes=1,
    activation='sigmoid'
)

# Load SimCLR weights
simclr_weights = torch.load(config.output_dir / "simclr_encoder.pth", map_location='cpu')
unet.encoder.load_state_dict(simclr_weights)
print("Loaded SimCLR pretrained encoder!")

# Reset BatchNorm
for m in unet.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.reset_running_stats()

unet = unet.to(config.device)

# Optimizer with differential LR
encoder_params = [p for n, p in unet.named_parameters() if 'encoder' in n]
decoder_params = [p for n, p in unet.named_parameters() if 'encoder' not in n]
optimizer = optim.AdamW([
    {"params": encoder_params, "lr": config.encoder_lr},
    {"params": decoder_params, "lr": config.decoder_lr}
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.finetune_epochs)

# Loss
mse_loss = nn.MSELoss()

# Training
best_ssim = 0
from skimage.metrics import structural_similarity as ssim

for epoch in range(config.finetune_epochs):
    unet.train()
    train_loss = 0
    
    for batch in tqdm(train_loader, desc=f"FT Epoch {epoch+1}"):
        inp = batch["input"].to(config.device)
        tgt = batch["target"].to(config.device)
        
        out = unet(inp)
        loss = mse_loss(out, tgt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    unet.eval()
    val_ssim = 0
    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input"].to(config.device)
            tgt = batch["target"].to(config.device)
            out = unet(inp)
            
            for i in range(out.shape[0]):
                p = out[i, 0].cpu().numpy()
                t = tgt[i, 0].cpu().numpy()
                val_ssim += ssim(t, p, data_range=1.0)
    
    val_ssim /= len(val_ds)
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, SSIM={val_ssim:.4f}")
    
    if val_ssim > best_ssim:
        best_ssim = val_ssim
        torch.save(unet.state_dict(), config.output_dir / "best_model.pth")

print(f"\n🎉 Best Val SSIM: {best_ssim:.4f}")
print(f"Model saved: {config.output_dir / 'best_model.pth'}")
