"""
Quick Validation Script for Medical AI Competition
- 1 fold only
- Category C only validation
- 3 epochs
- Purpose: Fast sanity check before running full CI

Rule: If this doesn't improve C's SSIM, don't bother with full CI.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Config:
    # Paths (adjust for local vs Kaggle)
    data_dir = Path(os.environ.get("DATA_DIR", "medical-ai-contest-7th-2025"))
    train_csv = data_dir / "train.csv"
    output_dir = Path(os.environ.get("OUTPUT_DIR", "output"))
    
    # Image
    image_size = 512
    in_channels = 1
    out_channels = 1
    
    # Quick training settings
    epochs = 3  # Fast
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_workers = 0  # For local testing
    
    # Loss weights
    l1_weight = 1.0
    ssim_weight = 2.0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


class OrganoidDataset(Dataset):
    def __init__(self, df, data_dir, image_size=512, category_filter=None):
        self.df = df.copy()
        if category_filter:
            self.df = self.df[self.df['category'].isin(category_filter)].reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        input_path = self.data_dir / row["input_path"]
        input_img = Image.open(input_path).convert("L")
        input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        input_arr = np.array(input_img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(input_arr).unsqueeze(0)
        
        target_path = self.data_dir / row["target_path"]
        target_img = Image.open(target_path).convert("L")
        target_img = target_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        target_arr = np.array(target_img, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(target_arr).unsqueeze(0)
        
        return {
            "id": row["id"], 
            "input": input_tensor, 
            "target": target_tensor,
            "category": row.get("category", "unknown")
        }


class SimpleUNet(nn.Module):
    """Minimal UNet for quick testing"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = self._double_conv(in_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._double_conv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
    
    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel != self.channel or self.window.data.type() != img1.data.type():
            self.window = self._create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.channel = channel
        
        mu1 = nn.functional.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = nn.functional.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = nn.functional.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=1.0):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        return self.l1_weight * self.l1_loss(pred, target) + self.ssim_weight * self.ssim_loss(pred, target)


def quick_validate_c(config):
    """
    Quick validation focusing on Category C only.
    Train: A + B + 80% of C
    Val: 20% of C (hardest samples ideally)
    """
    print("="*60)
    print("QUICK VALIDATION: Category C Focus")
    print("="*60)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    df = pd.read_csv(config.train_csv)
    print(f"Total samples: {len(df)}")
    
    # Split C into train/val (80/20)
    df_c = df[df['category'] == 'C'].reset_index(drop=True)
    n_val = len(df_c) // 5  # 20% for validation
    
    # Use last 20% as validation (could be improved with difficulty-based selection)
    val_c_idx = df_c.index[-n_val:]
    train_c_idx = df_c.index[:-n_val]
    
    # Training: A + B + 80% of C
    df_train = pd.concat([
        df[df['category'] == 'A'],
        df[df['category'] == 'B'],
        df_c.iloc[train_c_idx]
    ]).reset_index(drop=True)
    
    # Validation: 20% of C only
    df_val = df_c.iloc[val_c_idx].reset_index(drop=True)
    
    print(f"Train: {len(df_train)} (A+B + 80% C)")
    print(f"Val: {len(df_val)} (20% C only)")
    
    train_dataset = OrganoidDataset(df_train, config.data_dir, config.image_size)
    val_dataset = OrganoidDataset(df_val, config.data_dir, config.image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    model = SimpleUNet(config.in_channels, config.out_channels).to(config.device)
    criterion = CombinedLoss(config.l1_weight, config.ssim_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} Train"):
            inputs = batch["input"].to(config.device)
            targets = batch["target"].to(config.device)
            optimizer.zero_grad()
            outputs = torch.clamp(model(inputs), 0, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate on C only
        model.eval()
        ssim_scores = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} Val"):
                inputs = batch["input"].to(config.device)
                targets = batch["target"].to(config.device)
                outputs = torch.clamp(model(inputs), 0, 1)
                
                pred_np = outputs.cpu().numpy()
                target_np = targets.cpu().numpy()
                for i in range(pred_np.shape[0]):
                    p = np.clip(pred_np[i, 0], 0, 1)
                    t = np.clip(target_np[i, 0], 0, 1)
                    ssim_scores.append(ssim(t, p, data_range=1.0))
        
        mean_ssim = np.mean(ssim_scores)
        worst_20 = np.mean(sorted(ssim_scores)[:max(1, len(ssim_scores)//5)])
        
        print(f"Epoch {epoch+1}: C_val SSIM={mean_ssim:.4f}, C_worst20={worst_20:.4f}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"QUICK VALIDATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Final C SSIM:        {mean_ssim:.4f}")
    print(f"Final C worst 20%:   {worst_20:.4f}  ← This must improve!")
    print(f"{'='*60}")
    
    return {
        "ssim_C": mean_ssim,
        "ssim_C_worst20": worst_20,
        "time_seconds": elapsed,
    }


if __name__ == "__main__":
    config = Config()
    config.output_dir.mkdir(exist_ok=True, parents=True)
    results = quick_validate_c(config)
    
    with open(config.output_dir / "quick_validate_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {config.output_dir / 'quick_validate_results.json'}")
