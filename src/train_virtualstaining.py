"""
VirtualStaining Training Script
Train U-Net model with MSE+SSIM loss and CLAHE preprocessing
Dataset: Figshare DOI: 10.6084/m9.figshare.21971558
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Try to import segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
    USE_SMP = True
except ImportError:
    USE_SMP = False
    print("Warning: segmentation_models_pytorch not found.")


# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    # Data
    data_dir = Path("data/VirtualStaining")
    image_size = 512
    in_channels = 1
    out_channels = 1
    
    # Training
    epochs = 50
    batch_size = 4
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_workers = 0  # Windows compatibility
    
    # Loss weights
    mse_weight = 1.0
    ssim_weight = 1.0
    
    # CLAHE preprocessing
    use_clahe = True
    clahe_clip_limit = 2.0
    clahe_tile_size = (8, 8)
    
    # Model
    encoder = "resnet34"
    encoder_weights = "imagenet"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output
    output_dir = Path("outputs/virtualstaining")
    
    # Seed
    seed = 42


# ==============================================================================
# CLAHE Preprocessing
# ==============================================================================

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply CLAHE to grayscale image."""
    if len(image.shape) == 3:
        image = image[:, :, 0] if image.shape[2] == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


# ==============================================================================
# Dataset
# ==============================================================================

class VirtualStainingDataset(Dataset):
    """Dataset for VirtualStaining (phase-contrast → fluorescence)."""
    
    def __init__(
        self,
        data_dir: Path,
        image_size: int = 512,
        is_test: bool = False,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8)
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.is_test = is_test
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        
        # Find image pairs
        self.pairs = self._find_image_pairs()
        print(f"Found {len(self.pairs)} image pairs")
    
    def _find_image_pairs(self):
        """Find phase-contrast and fluorescence image pairs.
        
        VirtualStaining dataset format:
        - X.0.jpg = phase-contrast (input)
        - X.1.jpg = fluorescence channel 1 (target)
        - X.2.jpg = fluorescence channel 2 (optional)
        """
        pairs = []
        
        # Recursively find all image files
        all_images = list(self.data_dir.rglob("*.jpg")) + list(self.data_dir.rglob("*.png"))
        
        # Group by base number (e.g., "1.0.jpg" -> "1", "1.1.jpg" -> "1")
        triplets = {}
        for img_path in all_images:
            stem = img_path.stem  # e.g., "1.0" or "1.1" or "1.2"
            parts = stem.rsplit('.', 1)
            if len(parts) == 2:
                base_num, channel = parts
                key = (img_path.parent, base_num)  # Group by directory + base number
                if key not in triplets:
                    triplets[key] = {}
                triplets[key][channel] = img_path
        
        # Create pairs: phase (channel 0) -> fluorescence (channel 1)
        for (parent, base_num), channels in triplets.items():
            if '0' in channels and '1' in channels:
                pairs.append((channels['0'], channels['1']))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        phase_path, fluor_path = self.pairs[idx]
        
        # Load phase-contrast image (input)
        phase_img = cv2.imread(str(phase_path), cv2.IMREAD_GRAYSCALE)
        if phase_img is None:
            phase_img = np.array(Image.open(phase_path).convert("L"))
        
        # Apply CLAHE to input
        if self.use_clahe:
            phase_img = apply_clahe(phase_img, self.clahe_clip_limit, self.clahe_tile_size)
        
        # Resize
        phase_img = cv2.resize(phase_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        phase_arr = phase_img.astype(np.float32) / 255.0
        phase_tensor = torch.from_numpy(phase_arr).unsqueeze(0)  # (1, H, W)
        
        if self.is_test:
            return {
                "id": phase_path.stem,
                "input": phase_tensor,
            }
        
        # Load fluorescence image (target)
        fluor_img = cv2.imread(str(fluor_path), cv2.IMREAD_GRAYSCALE)
        if fluor_img is None:
            fluor_img = np.array(Image.open(fluor_path).convert("L"))
        
        fluor_img = cv2.resize(fluor_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        fluor_arr = fluor_img.astype(np.float32) / 255.0
        fluor_tensor = torch.from_numpy(fluor_arr).unsqueeze(0)  # (1, H, W)
        
        return {
            "id": phase_path.stem,
            "input": phase_tensor,
            "target": fluor_tensor,
        }


# ==============================================================================
# Loss Functions
# ==============================================================================

class SSIMLoss(nn.Module):
    """Differentiable SSIM Loss."""
    
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
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
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel != self.channel or self.window.data.type() != img1.data.type():
            self.window = self._create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.channel = channel
        
        mu1 = nn.functional.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = nn.functional.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = nn.functional.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class MSESSIMLoss(nn.Module):
    """Combined MSE + SSIM Loss."""
    
    def __init__(self, mse_weight=1.0, ssim_weight=1.0):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.mse_weight * mse + self.ssim_weight * ssim


# ==============================================================================
# Simple U-Net (fallback)
# ==============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ==============================================================================
# Training Functions
# ==============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(pred, target):
    """Calculate SSIM and PSNR for a batch."""
    from skimage.metrics import structural_similarity as ssim_func
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    ssim_scores = []
    psnr_scores = []
    
    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i, 0], 0, 1)
        t = np.clip(target_np[i, 0], 0, 1)
        
        ssim_val = ssim_func(t, p, data_range=1.0)
        psnr_val = psnr_func(t, p, data_range=1.0)
        
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)
    
    return np.mean(ssim_scores), np.mean(psnr_scores)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.clamp(outputs, 0, 1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_ssim = 0
    total_psnr = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            batch_ssim, batch_psnr = calculate_metrics(outputs, targets)
            total_ssim += batch_ssim
            total_psnr += batch_psnr
            n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "ssim": total_ssim / n_batches,
        "psnr": total_psnr / n_batches,
    }


def train(config):
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    full_dataset = VirtualStainingDataset(
        config.data_dir,
        config.image_size,
        is_test=False,
        use_clahe=config.use_clahe,
        clahe_clip_limit=config.clahe_clip_limit,
        clahe_tile_size=config.clahe_tile_size
    )
    
    if len(full_dataset) == 0:
        print("ERROR: No image pairs found. Please check data directory structure.")
        print(f"Expected data in: {config.data_dir}")
        return None, []
    
    # Train/Val split
    n_samples = len(full_dataset)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Model
    if USE_SMP:
        model = smp.Unet(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.out_channels,
        )
    else:
        model = SimpleUNet(config.in_channels, config.out_channels)
    
    model = model.to(config.device)
    print(f"Model: {'SMP U-Net' if USE_SMP else 'Simple U-Net'}")
    print(f"Device: {config.device}")
    
    # Loss and optimizer
    criterion = MSESSIMLoss(config.mse_weight, config.ssim_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    
    # Training loop
    best_ssim = 0
    history = []
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_metrics = validate(model, val_loader, criterion, config.device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, SSIM: {val_metrics['ssim']:.4f}, PSNR: {val_metrics['psnr']:.2f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_ssim": val_metrics["ssim"],
            "val_psnr": val_metrics["psnr"],
        })
        
        if val_metrics["ssim"] > best_ssim:
            best_ssim = val_metrics["ssim"]
            torch.save(model.state_dict(), config.output_dir / "best_model.pth")
            print(f"Saved best model (SSIM: {best_ssim:.4f})")
    
    # Save metrics
    metrics = {
        "best_ssim": float(best_ssim),
        "best_psnr": float(max(h["val_psnr"] for h in history)),
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "mse_weight": config.mse_weight,
            "ssim_weight": config.ssim_weight,
            "use_clahe": config.use_clahe,
        },
        "history": history
    }
    
    with open(config.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Best SSIM: {best_ssim:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="VirtualStaining Training with MSE+SSIM Loss")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--data-dir", type=str, default="data/VirtualStaining")
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=1.0)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--quick-check", action="store_true")
    args = parser.parse_args()
    
    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.image_size = args.image_size
    config.data_dir = Path(args.data_dir)
    config.mse_weight = args.mse_weight
    config.ssim_weight = args.ssim_weight
    config.use_clahe = not args.no_clahe
    
    if args.quick_check:
        config.epochs = 1
        config.batch_size = 2
    
    train(config)


if __name__ == "__main__":
    main()
