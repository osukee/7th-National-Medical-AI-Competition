"""
Medical AI Competition Training Script v2 - Mask-Based Baseline
Key fixes:
1. Loss computed on MASK region only
2. Evaluation uses SSIM(mask) + Normalized_PSNR(mask) / 2
3. epochs=15 (not 3)
4. Simplified: no C-weighting, no clustering (add after baseline works)
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
from sklearn.model_selection import StratifiedKFold
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
    
    # Training - KEY CHANGE: epochs=30 for convergence check
    epochs = 30
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_workers = 2
    
    # Loss weights - As suggested: SSIM + 0.1*L1 + 0.02*MSE
    ssim_weight = 1.0
    l1_weight = 0.1
    mse_weight = 0.02
    
    # Model
    encoder = "resnet34"
    encoder_weights = "imagenet"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seed
    seed = 42


# ==============================================================================
# Dataset with Mask
# ==============================================================================

class OrganoidDatasetWithMask(Dataset):
    """Dataset that loads input, target, AND mask for mask-based training."""
    
    def __init__(self, csv_path_or_df, data_dir, image_size=512, is_test=False, indices=None):
        if isinstance(csv_path_or_df, pd.DataFrame):
            self.df = csv_path_or_df.copy()
        else:
            self.df = pd.read_csv(csv_path_or_df)
        
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
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
        
        # Load mask - KEY ADDITION
        mask_path = self.data_dir / row["mask_path"]
        mask_img = Image.open(mask_path).convert("L")
        # NEAREST interpolation for mask to avoid creating intermediate values
        mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_arr = np.array(mask_img, dtype=np.float32)
        # Binarize: >127 = 1.0, else 0.0
        mask_arr = (mask_arr > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        
        category = row.get("category", "unknown")
        
        return {
            "id": row["id"], 
            "input": input_tensor, 
            "target": target_tensor, 
            "mask": mask_tensor,
            "category": category,
        }


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
# Mask-Based Loss Functions
# ==============================================================================

class MaskedSSIMLoss(nn.Module):
    """SSIM Loss computed only on masked regions."""
    
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
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W)
            mask: (B, 1, H, W) binary mask
        """
        (_, channel, _, _) = pred.size()
        
        if channel != self.channel or self.window.data.type() != pred.data.type():
            self.window = self._create_window(self.window_size, channel).to(pred.device).type(pred.dtype)
            self.channel = channel
        
        mu1 = nn.functional.conv2d(pred, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = nn.functional.conv2d(target, self.window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = nn.functional.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(target * target, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Apply mask: only compute loss on masked regions
        ssim_map_masked = ssim_map * mask
        
        # Normalize by mask sum (avoid division by zero)
        mask_sum = mask.sum()
        if mask_sum > 0:
            ssim_val = ssim_map_masked.sum() / mask_sum
        else:
            ssim_val = ssim_map.mean()
        
        return 1 - ssim_val


class MaskedL1Loss(nn.Module):
    """L1 Loss computed only on masked regions."""
    
    def forward(self, pred, target, mask):
        diff = torch.abs(pred - target)
        masked_diff = diff * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            return masked_diff.sum() / mask_sum
        return diff.mean()


class MaskedMSELoss(nn.Module):
    """MSE Loss computed only on masked regions."""
    
    def forward(self, pred, target, mask):
        diff_sq = (pred - target) ** 2
        masked_diff = diff_sq * mask
        mask_sum = mask.sum()
        if mask_sum > 0:
            return masked_diff.sum() / mask_sum
        return diff_sq.mean()


class MaskedCombinedLoss(nn.Module):
    """Combined Loss: SSIM + L1 + MSE, all masked."""
    
    def __init__(self, ssim_weight=1.0, l1_weight=0.1, mse_weight=0.02):
        super().__init__()
        self.ssim_loss = MaskedSSIMLoss()
        self.l1_loss = MaskedL1Loss()
        self.mse_loss = MaskedMSELoss()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
    
    def forward(self, pred, target, mask):
        ssim = self.ssim_loss(pred, target, mask)
        l1 = self.l1_loss(pred, target, mask)
        mse = self.mse_loss(pred, target, mask)
        return self.ssim_weight * ssim + self.l1_weight * l1 + self.mse_weight * mse


# ==============================================================================
# Mask-Based Evaluation Metrics
# ==============================================================================

def calculate_metrics_masked(pred, target, mask):
    """
    Calculate SSIM and PSNR on masked regions only.
    
    Args:
        pred: (B, 1, H, W) tensor, values 0-1
        target: (B, 1, H, W) tensor, values 0-1
        mask: (B, 1, H, W) tensor, binary
    
    Returns:
        dict with ssim, psnr, psnr_normalized, score (per batch, averaged)
    """
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    mask_np = mask.cpu().numpy()
    
    ssim_scores = []
    psnr_scores = []
    
    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i, 0], 0, 1) * 255  # Convert to 0-255
        t = np.clip(target_np[i, 0], 0, 1) * 255
        m = mask_np[i, 0] > 0.5  # Boolean mask
        
        if m.sum() == 0:
            continue
        
        # Extract masked pixels
        p_m = p[m]
        t_m = t[m]
        
        # SSIM on masked region (simplified global)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        mu_p, mu_t = p_m.mean(), t_m.mean()
        s_p, s_t = p_m.var(), t_m.var()
        s_pt = ((p_m - mu_p) * (t_m - mu_t)).mean()
        ssim_val = ((2*mu_p*mu_t + C1) * (2*s_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (s_p + s_t + C2))
        ssim_scores.append(ssim_val)
        
        # PSNR on masked region
        mse = np.mean((p_m - t_m) ** 2)
        if mse == 0:
            psnr_val = 100.0
        else:
            psnr_val = 10 * np.log10((255 ** 2) / mse)
        psnr_scores.append(psnr_val)
    
    if not ssim_scores:
        return {'ssim': 0.0, 'psnr': 0.0, 'psnr_normalized': 0.0, 'score': 0.0}
    
    ssim_mean = np.mean(ssim_scores)
    psnr_mean = np.mean(psnr_scores)
    psnr_norm = np.clip((psnr_mean - 15) / 20, 0, 1)
    score = (ssim_mean + psnr_norm) / 2
    
    return {
        'ssim': float(ssim_mean),
        'psnr': float(psnr_mean),
        'psnr_normalized': float(psnr_norm),
        'score': float(score),
    }


# ==============================================================================
# Training Functions
# ==============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        outputs = torch.clamp(model(inputs), 0, 1)
        
        loss = criterion(outputs, targets, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            masks = batch["mask"].to(device)
            
            outputs = torch.clamp(model(inputs), 0, 1)
            
            loss = criterion(outputs, targets, masks)
            total_loss += loss.item()
            
            # Calculate masked metrics
            batch_metrics = calculate_metrics_masked(outputs, targets, masks)
            all_metrics.append(batch_metrics)
    
    # Aggregate
    avg_metrics = {
        'loss': total_loss / len(loader),
        'ssim': np.mean([m['ssim'] for m in all_metrics]),
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'psnr_normalized': np.mean([m['psnr_normalized'] for m in all_metrics]),
        'score': np.mean([m['score'] for m in all_metrics]),
    }
    
    return avg_metrics


# ==============================================================================
# Main Training
# ==============================================================================

def train_simple(config):
    """Simple training without K-Fold. For baseline verification."""
    start_time = time.time()
    set_seed(config.seed)
    
    print("=" * 60)
    print("Mask-Based Baseline Training v2")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Loss weights: SSIM={config.ssim_weight}, L1={config.l1_weight}, MSE={config.mse_weight}")
    
    # Load data
    df = pd.read_csv(config.train_csv)
    print(f"Total samples: {len(df)}")
    
    # Simple train/val split (80/20)
    n_samples = len(df)
    indices = np.random.RandomState(config.seed).permutation(n_samples)
    n_train = int(n_samples * 0.8)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_dataset = OrganoidDatasetWithMask(df, config.data_dir, config.image_size, indices=train_idx)
    val_dataset = OrganoidDatasetWithMask(df, config.data_dir, config.image_size, indices=val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = create_model(config)
    
    # Loss - masked combined
    criterion = MaskedCombinedLoss(config.ssim_weight, config.l1_weight, config.mse_weight)
    
    # Optimizer with cosine schedule
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    
    # Training loop
    best_score = 0
    history = []
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_metrics = validate(model, val_loader, criterion, config.device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val SSIM (mask): {val_metrics['ssim']:.4f}")
        print(f"Val PSNR (mask): {val_metrics['psnr']:.2f} dB")
        print(f"Val Score: {val_metrics['score']:.4f}  ← Competition metric")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_ssim": val_metrics["ssim"],
            "val_psnr": val_metrics["psnr"],
            "val_score": val_metrics["score"],
        })
        
        # Save best by SCORE (not just SSIM)
        if val_metrics["score"] > best_score:
            best_score = val_metrics["score"]
            torch.save(model.state_dict(), config.output_dir / "best_model.pth")
            print(f"🎯 Saved best model (Score: {best_score:.4f})")
    
    training_time = time.time() - start_time
    
    # Final metrics
    best_history = max(history, key=lambda x: x["val_score"])
    
    final_metrics = {
        "experiment_id": os.environ.get("EXPERIMENT_ID", "baseline_v2"),
        "timestamp": datetime.now().isoformat(),
        "cv_mode": "simple_split",
        "metrics": {
            "ssim": float(best_history["val_ssim"]),
            "psnr": float(best_history["val_psnr"]),
            "score": float(best_history["val_score"]),
        },
        "training_time_seconds": int(training_time),
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "image_size": config.image_size,
            "ssim_weight": config.ssim_weight,
            "l1_weight": config.l1_weight,
            "mse_weight": config.mse_weight,
        }
    }
    
    with open(config.output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    with open(config.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Score: {best_score:.4f}")
    print(f"  SSIM: {best_history['val_ssim']:.4f}")
    print(f"  PSNR: {best_history['val_psnr']:.2f} dB")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    return model, history


# ==============================================================================
# Inference
# ==============================================================================

def predict_and_submit(config, model_path=None):
    """Run inference on test set and create submission CSV."""
    print(f"\n{'='*60}")
    print("Running Inference on Test Set")
    print(f"{'='*60}")
    
    import cv2
    
    model = create_model(config)
    
    if model_path is None:
        model_path = config.output_dir / "best_model.pth"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    test_df = pd.read_csv(config.test_csv)
    print(f"Test samples: {len(test_df)}")
    
    test_dataset = OrganoidDatasetWithMask(test_df, config.data_dir, config.image_size, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    all_ids = []
    all_pixels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            inputs = batch["input"].to(config.device)
            ids = batch["id"]
            
            outputs = torch.clamp(model(inputs), 0, 1)
            
            for i, sample_id in enumerate(ids):
                pred = outputs[i, 0].cpu().numpy()
                pred_uint8 = (pred * 255).astype(np.uint8)
                
                if pred_uint8.shape != (512, 512):
                    pred_uint8 = cv2.resize(pred_uint8, (512, 512))
                
                pixels_flat = pred_uint8.flatten()
                
                all_ids.append(sample_id)
                all_pixels.append(pixels_flat)
    
    print(f"\nCreating submission CSV...")
    print(f"Number of samples: {len(all_ids)}")
    
    if all_pixels:
        first_pixels = all_pixels[0]
        print(f"First sample: min={first_pixels.min()}, max={first_pixels.max()}, mean={first_pixels.mean():.1f}")
    
    n_pixels = 512 * 512
    pixel_columns = [f"pixel_{i}" for i in range(n_pixels)]
    
    submission_df = pd.DataFrame(all_pixels, columns=pixel_columns)
    submission_df.insert(0, "id", all_ids)
    
    csv_path = config.output_dir / "submission.csv"
    submission_df.to_csv(csv_path, index=False)
    
    print(f"📄 Submission saved: {csv_path}")
    print(f"   Shape: {submission_df.shape}")
    
    return csv_path


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    config = Config()
    
    # Allow env overrides
    if os.environ.get("EPOCHS"):
        config.epochs = int(os.environ["EPOCHS"])
    if os.environ.get("BATCH_SIZE"):
        config.batch_size = int(os.environ["BATCH_SIZE"])
    
    run_inference = os.environ.get("RUN_INFERENCE", "1") == "1"
    
    # Train
    train_simple(config)
    
    # Inference
    if run_inference:
        predict_and_submit(config)
