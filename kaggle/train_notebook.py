"""
Medical AI Competition Training Script for Kaggle
This script is designed to run on Kaggle's GPU environment.
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

# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    # Kaggle paths
    data_dir = Path("/kaggle/input/medical-ai-contest-7th-2025")
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    output_dir = Path("/kaggle/working")
    
    # Image
    image_size = 512
    in_channels = 1
    out_channels = 1
    
    # Training
    epochs = 10  # 50→10: タイムアウト回避のため削減（仮説検証）
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_workers = 2
    
    # Loss weights
    l1_weight = 1.0
    ssim_weight = 1.0
    
    # Model
    encoder = "resnet34"
    encoder_weights = "imagenet"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seed
    seed = 42


# ==============================================================================
# Dataset
# ==============================================================================

class OrganoidDataset(Dataset):
    def __init__(self, csv_path, data_dir, image_size=512, is_test=False):
        self.df = pd.read_csv(csv_path)
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load input image
        input_path = self.data_dir / row["input_path"]
        input_img = Image.open(input_path).convert("L")
        input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        input_arr = np.array(input_img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(input_arr).unsqueeze(0)
        
        if self.is_test:
            return {"id": row["id"], "input": input_tensor}
        
        # Load target image
        target_path = self.data_dir / row["target_path"]
        target_img = Image.open(target_path).convert("L")
        target_img = target_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        target_arr = np.array(target_img, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(target_arr).unsqueeze(0)
        
        return {"id": row["id"], "input": input_tensor, "target": target_tensor}


# ==============================================================================
# Model
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
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
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


def create_model(config):
    """Create model, preferring SMP if available."""
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.out_channels,
        )
        print("Using SMP U-Net with pretrained encoder")
    except ImportError:
        model = SimpleUNet(config.in_channels, config.out_channels)
        print("Using Simple U-Net (SMP not available)")
    
    return model.to(config.device)


# ==============================================================================
# Loss Functions
# ==============================================================================

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
        
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=1.0):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.l1_weight * l1 + self.ssim_weight * ssim


# ==============================================================================
# Training
# ==============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(pred, target):
    """Calculate SSIM and PSNR for a batch."""
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    ssim_scores = []
    psnr_scores = []
    
    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i, 0], 0, 1)
        t = np.clip(target_np[i, 0], 0, 1)
        
        ssim_scores.append(ssim(t, p, data_range=1.0))
        psnr_scores.append(psnr(t, p, data_range=1.0))
    
    return np.mean(ssim_scores), np.mean(psnr_scores)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        
        optimizer.zero_grad()
        outputs = torch.clamp(model(inputs), 0, 1)
        
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
            
            outputs = torch.clamp(model(inputs), 0, 1)
            
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
    start_time = time.time()
    set_seed(config.seed)
    
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    
    # Dataset
    full_dataset = OrganoidDataset(
        config.train_csv,
        config.data_dir,
        config.image_size,
        is_test=False
    )
    
    # Train/Val split
    n_samples = len(full_dataset)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Model
    model = create_model(config)
    
    # Loss and optimizer
    criterion = CombinedLoss(config.l1_weight, config.ssim_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
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
        
        # Save best model
        if val_metrics["ssim"] > best_ssim:
            best_ssim = val_metrics["ssim"]
            torch.save(model.state_dict(), config.output_dir / "best_model.pth")
            print(f"Saved best model (SSIM: {best_ssim:.4f})")
    
    training_time = time.time() - start_time
    
    # Save final metrics for GitHub Actions to retrieve
    final_metrics = {
        "experiment_id": os.environ.get("EXPERIMENT_ID", "kaggle_run"),
        "timestamp": datetime.now().isoformat(),
        "commit_sha": os.environ.get("COMMIT_SHA", "unknown"),
        "branch": os.environ.get("BRANCH_NAME", "unknown"),
        "metrics": {
            "ssim": float(best_ssim),
            "psnr": float(history[-1]["val_psnr"]),
            "ssim_std": float(np.std([h["val_ssim"] for h in history[-5:]])) if len(history) >= 5 else 0.0,
            "psnr_std": float(np.std([h["val_psnr"] for h in history[-5:]])) if len(history) >= 5 else 0.0,
        },
        "training_time_seconds": int(training_time),
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "image_size": config.image_size,
        }
    }
    
    with open(config.output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save training history
    with open(config.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best SSIM: {best_ssim:.4f}")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"{'='*50}")
    
    return model, history


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    config = Config()
    
    # Allow overrides from environment variables
    if os.environ.get("EPOCHS"):
        config.epochs = int(os.environ["EPOCHS"])
    if os.environ.get("BATCH_SIZE"):
        config.batch_size = int(os.environ["BATCH_SIZE"])
    if os.environ.get("LEARNING_RATE"):
        config.learning_rate = float(os.environ["LEARNING_RATE"])
    
    train(config)
