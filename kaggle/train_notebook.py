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
    
    # Training
    epochs = 3  # 50→3: ベースライン動作確認用（通過後に増やす）
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
# Dark Ratio Computation (Continuous, No Clustering)
# ==============================================================================

def compute_dark_ratio(df, data_dir):
    """
    Compute dark_ratio for all samples as a continuous difficulty measure.
    No discrete clustering - this avoids boundary artifacts.
    
    Returns df with 'dark_ratio' column added.
    """
    print("Computing dark_ratio for all samples...")
    
    df = df.copy()
    df['dark_ratio'] = 0.0
    
    for idx, row in df.iterrows():
        try:
            input_path = Path(data_dir) / row['input_path']
            img = Image.open(input_path).convert('L')
            arr = np.array(img)
            
            # dark_ratio: percentage of very dark pixels (< 50)
            dark_ratio = (arr < 50).sum() / arr.size
            df.loc[idx, 'dark_ratio'] = dark_ratio
        except Exception:
            df.loc[idx, 'dark_ratio'] = 0.0
    
    print(f"Dark ratio stats: mean={df['dark_ratio'].mean():.3f}, "
          f"std={df['dark_ratio'].std():.3f}, "
          f"min={df['dark_ratio'].min():.3f}, max={df['dark_ratio'].max():.3f}")
    
    return df


def create_worst_case_splits(df, data_dir, worst_val_ratio=0.20, c_hard_train_ratio=0.60):
    """
    Create worst-case controlled CV splits.
    
    Strategy:
    1. Sort all samples by dark_ratio (continuous, no boundaries)
    2. Top 20% of C (by dark_ratio) → worst_val (fixed across all folds)
    3. Next 40% of C_hard → Train fixed (60%)
    4. Remaining samples → Normal Stratified K-Fold
    
    Returns:
        - worst_val_idx: Fixed validation indices for worst-case (evaluated every fold)
        - trainable_idx: Indices available for K-Fold splitting
        - c_hard_train_idx: C_hard samples fixed in train
    """
    # Compute dark_ratio if not already present
    if 'dark_ratio' not in df.columns:
        df = compute_dark_ratio(df, data_dir)
    
    # Get Category C samples sorted by dark_ratio (descending = harder first)
    c_mask = df['category'] == 'C'
    df_c = df[c_mask].sort_values('dark_ratio', ascending=False)
    
    n_c = len(df_c)
    n_worst = int(n_c * worst_val_ratio)  # top 20% = ~80 samples
    n_c_hard = int(n_c * 0.40)  # next 40% after worst = ~160 samples
    n_c_hard_train = int(n_c_hard * c_hard_train_ratio)  # 60% of C_hard → train fixed
    
    # Split C samples
    worst_val_idx = df_c.index[:n_worst].tolist()
    c_hard_idx = df_c.index[n_worst:n_worst + n_c_hard].tolist()
    c_hard_train_idx = c_hard_idx[:n_c_hard_train]
    c_hard_foldable_idx = c_hard_idx[n_c_hard_train:]
    c_normal_idx = df_c.index[n_worst + n_c_hard:].tolist()
    
    # Get A, B samples
    ab_idx = df[~c_mask].index.tolist()
    
    # Trainable = A, B, C_normal, C_hard foldable (not worst_val, not c_hard_train)
    trainable_idx = ab_idx + c_normal_idx + c_hard_foldable_idx
    
    print(f"\nWorst-Case Split Summary:")
    print(f"  worst_val (fixed):      {len(worst_val_idx)} samples (C dark_ratio top {worst_val_ratio*100:.0f}%)")
    print(f"  c_hard_train (fixed):   {len(c_hard_train_idx)} samples (60% of C_hard)")
    print(f"  trainable (K-Fold):     {len(trainable_idx)} samples")
    print(f"  Total:                  {len(worst_val_idx) + len(c_hard_train_idx) + len(trainable_idx)}")
    
    return {
        'worst_val_idx': worst_val_idx,
        'c_hard_train_idx': c_hard_train_idx,
        'trainable_idx': trainable_idx,
        'df': df,  # df with dark_ratio column
    }


def create_worst_case_splits_v5(df, data_dir):
    """
    v5: Worst-case splits with worst_val split into train and eval.
    
    Key changes from v4:
    - worst_train (top 10% of C): Goes to Train with Loss×3
    - worst_eval (10-20% of C): Evaluation only
    - c_hard_train: Train with Loss×2
    
    Returns dict with sample weights for loss weighting.
    """
    # Compute dark_ratio if not already present
    if 'dark_ratio' not in df.columns:
        df = compute_dark_ratio(df, data_dir)
    
    # Get Category C samples sorted by dark_ratio (descending = harder first)
    c_mask = df['category'] == 'C'
    df_c = df[c_mask].sort_values('dark_ratio', ascending=False)
    
    n_c = len(df_c)
    n_worst_train = int(n_c * 0.10)  # top 10% → Train with Loss×3
    n_worst_eval = int(n_c * 0.10)   # next 10% → Eval only
    n_c_hard = int(n_c * 0.30)       # next 30% → C_hard
    n_c_hard_train = int(n_c_hard * 0.60)  # 60% of C_hard → Train with Loss×2
    
    # Split C samples
    worst_train_idx = df_c.index[:n_worst_train].tolist()
    worst_eval_idx = df_c.index[n_worst_train:n_worst_train + n_worst_eval].tolist()
    c_hard_idx = df_c.index[n_worst_train + n_worst_eval:n_worst_train + n_worst_eval + n_c_hard].tolist()
    c_hard_train_idx = c_hard_idx[:n_c_hard_train]
    c_hard_foldable_idx = c_hard_idx[n_c_hard_train:]
    c_normal_idx = df_c.index[n_worst_train + n_worst_eval + n_c_hard:].tolist()
    
    # Get A, B samples
    ab_idx = df[~c_mask].index.tolist()
    
    # Trainable = A, B, C_normal, C_hard foldable
    trainable_idx = ab_idx + c_normal_idx + c_hard_foldable_idx
    
    # Create sample weight mapping (for loss weighting)
    sample_weights = {}
    for idx in worst_train_idx:
        sample_weights[idx] = 3.0  # worst_train: ×3
    for idx in c_hard_train_idx:
        sample_weights[idx] = 2.0  # c_hard_train: ×2
    # Others default to 1.0
    
    print(f"\nv5 Worst-Case Split Summary:")
    print(f"  worst_train (Train, Loss×3): {len(worst_train_idx)} samples (C dark_ratio top 10%)")
    print(f"  worst_eval (Eval only):      {len(worst_eval_idx)} samples (C dark_ratio 10-20%)")
    print(f"  c_hard_train (Train, Loss×2): {len(c_hard_train_idx)} samples")
    print(f"  trainable (K-Fold):           {len(trainable_idx)} samples")
    
    return {
        'worst_train_idx': worst_train_idx,
        'worst_eval_idx': worst_eval_idx,
        'c_hard_train_idx': c_hard_train_idx,
        'trainable_idx': trainable_idx,
        'sample_weights': sample_weights,
        'df': df,
    }

# ==============================================================================
# Dataset
# ==============================================================================

class OrganoidDataset(Dataset):
    def __init__(self, csv_path_or_df, data_dir, image_size=512, is_test=False, indices=None, sample_weights=None):
        if isinstance(csv_path_or_df, pd.DataFrame):
            self.df = csv_path_or_df.copy()
        else:
            self.df = pd.read_csv(csv_path_or_df)
        
        # Store original indices before reset
        if indices is not None:
            self.original_indices = indices
            self.df = self.df.loc[indices].reset_index(drop=True)
        else:
            self.original_indices = self.df.index.tolist()
        
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.is_test = is_test
        
        # Sample weights for loss weighting (v5)
        self.sample_weights = sample_weights or {}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        original_idx = self.original_indices[idx] if idx < len(self.original_indices) else idx
        
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
        
        # Include category for stratified evaluation
        category = row.get("category", "unknown")
        
        # Get sample weight (default 1.0)
        weight = self.sample_weights.get(original_idx, 1.0)
        
        return {
            "id": row["id"], 
            "input": input_tensor, 
            "target": target_tensor, 
            "category": category,
            "weight": torch.tensor(weight, dtype=torch.float32),
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


def train_epoch_weighted(model, loader, criterion, optimizer, device):
    """Training epoch with sample-wise loss weighting for v5."""
    model.train()
    total_loss = 0
    total_weighted_loss = 0
    
    pbar = tqdm(loader, desc="Training (weighted)")
    for batch in pbar:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        weights = batch["weight"].to(device)  # Sample weights
        
        optimizer.zero_grad()
        outputs = torch.clamp(model(inputs), 0, 1)
        
        # Compute per-sample loss and apply weights
        batch_size = inputs.size(0)
        sample_losses = []
        for i in range(batch_size):
            sample_loss = criterion(outputs[i:i+1], targets[i:i+1])
            sample_losses.append(sample_loss * weights[i])
        
        # Weighted mean loss
        weighted_loss = torch.stack(sample_losses).mean()
        weighted_loss.backward()
        optimizer.step()
        
        total_weighted_loss += weighted_loss.item()
        pbar.set_postfix({"w_loss": f"{weighted_loss.item():.4f}"})
    
    return total_weighted_loss / len(loader)


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


def validate_with_categories(model, loader, criterion, device):
    """Validation with category-wise metrics."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    category_metrics = {
        'A': {'ssim': [], 'psnr': []},
        'B': {'ssim': [], 'psnr': []},
        'C': {'ssim': [], 'psnr': []},
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            categories = batch["category"]
            
            outputs = torch.clamp(model(inputs), 0, 1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1
            
            # Calculate per-sample metrics and group by category
            pred_np = outputs.cpu().numpy()
            target_np = targets.cpu().numpy()
            
            for i, cat in enumerate(categories):
                p = np.clip(pred_np[i, 0], 0, 1)
                t = np.clip(target_np[i, 0], 0, 1)
                
                s = ssim(t, p, data_range=1.0)
                pn = psnr(t, p, data_range=1.0)
                
                if cat in category_metrics:
                    category_metrics[cat]['ssim'].append(s)
                    category_metrics[cat]['psnr'].append(pn)
    
    # Compute category-wise means
    results = {"loss": total_loss / n_batches}
    
    for cat in ['A', 'B', 'C']:
        if category_metrics[cat]['ssim']:
            results[f'ssim_{cat}'] = float(np.mean(category_metrics[cat]['ssim']))
            results[f'psnr_{cat}'] = float(np.mean(category_metrics[cat]['psnr']))
        else:
            results[f'ssim_{cat}'] = 0.0
            results[f'psnr_{cat}'] = 0.0
    
    # Worst-case metrics: bottom 20% of Category C (the real bottleneck)
    if category_metrics['C']['ssim']:
        c_ssim_sorted = sorted(category_metrics['C']['ssim'])
        c_psnr_sorted = sorted(category_metrics['C']['psnr'])
        n_worst = max(1, len(c_ssim_sorted) // 5)  # bottom 20%
        results['ssim_C_worst20'] = float(np.mean(c_ssim_sorted[:n_worst]))
        results['psnr_C_worst20'] = float(np.mean(c_psnr_sorted[:n_worst]))
    else:
        results['ssim_C_worst20'] = 0.0
        results['psnr_C_worst20'] = 0.0
    
    # Overall metrics (average across categories)
    valid_ssim = [results[f'ssim_{c}'] for c in ['A', 'B', 'C'] if results[f'ssim_{c}'] > 0]
    valid_psnr = [results[f'psnr_{c}'] for c in ['A', 'B', 'C'] if results[f'psnr_{c}'] > 0]
    
    results['ssim'] = float(np.mean(valid_ssim)) if valid_ssim else 0.0
    results['psnr'] = float(np.mean(valid_psnr)) if valid_psnr else 0.0
    
    return results


def train(config):
    """Original train function (single split, kept for backwards compatibility)."""
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


def train_kfold(config, n_folds=5):
    """Train using Stratified K-Fold Cross-Validation with difficulty-based stratification."""
    start_time = time.time()
    set_seed(config.seed)
    
    print(f"{'='*60}")
    print(f"Stratified {n_folds}-Fold Cross-Validation (by Difficulty)")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"Epochs per fold: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    
    # Load full dataframe
    df = pd.read_csv(config.train_csv)
    print(f"Total samples: {len(df)}")
    print(f"Category distribution: {df['category'].value_counts().to_dict()}")
    
    # Cluster Category C into difficulty sub-groups (A, B, C_easy, C_medium, C_hard)
    df = cluster_category_c(df, config.data_dir, n_clusters=3)
    
    # Stratified K-Fold by difficulty (not just category)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    
    fold_results = []
    best_overall_ssim = 0
    best_fold = -1
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['difficulty'])):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        print(f"{'='*60}")
        
        # Create datasets with indices
        train_dataset = OrganoidDataset(
            df, config.data_dir, config.image_size, is_test=False, indices=train_idx
        )
        val_dataset = OrganoidDataset(
            df, config.data_dir, config.image_size, is_test=False, indices=val_idx
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
        
        # Create fresh model for each fold
        model = create_model(config)
        
        criterion = CombinedLoss(config.l1_weight, config.ssim_weight)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
        
        # Training loop for this fold
        best_fold_ssim = 0
        
        for epoch in range(config.epochs):
            print(f"\nFold {fold+1} - Epoch {epoch + 1}/{config.epochs}")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
            val_metrics = validate_with_categories(model, val_loader, criterion, config.device)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val SSIM: {val_metrics['ssim']:.4f} (A:{val_metrics['ssim_A']:.4f}, B:{val_metrics['ssim_B']:.4f}, C:{val_metrics['ssim_C']:.4f})")
            print(f"Val PSNR: {val_metrics['psnr']:.2f} (A:{val_metrics['psnr_A']:.2f}, B:{val_metrics['psnr_B']:.2f}, C:{val_metrics['psnr_C']:.2f})")
            
            if val_metrics['ssim'] > best_fold_ssim:
                best_fold_ssim = val_metrics['ssim']
                # Save best model for this fold
                torch.save(model.state_dict(), config.output_dir / f"best_model_fold{fold}.pth")
        
        # Store fold results
        final_metrics = validate_with_categories(model, val_loader, criterion, config.device)
        fold_results.append({
            'fold': fold + 1,
            'ssim': final_metrics['ssim'],
            'psnr': final_metrics['psnr'],
            'ssim_A': final_metrics['ssim_A'],
            'ssim_B': final_metrics['ssim_B'],
            'ssim_C': final_metrics['ssim_C'],
            'ssim_C_worst20': final_metrics['ssim_C_worst20'],
            'psnr_A': final_metrics['psnr_A'],
            'psnr_B': final_metrics['psnr_B'],
            'psnr_C': final_metrics['psnr_C'],
            'psnr_C_worst20': final_metrics['psnr_C_worst20'],
        })
        
        if final_metrics['ssim'] > best_overall_ssim:
            best_overall_ssim = final_metrics['ssim']
            best_fold = fold
            # Save as overall best model
            torch.save(model.state_dict(), config.output_dir / "best_model.pth")
    
    training_time = time.time() - start_time
    
    # Aggregate results
    cv_results = {
        'n_folds': n_folds,
        'ssim_mean': float(np.mean([r['ssim'] for r in fold_results])),
        'ssim_std': float(np.std([r['ssim'] for r in fold_results])),
        'psnr_mean': float(np.mean([r['psnr'] for r in fold_results])),
        'psnr_std': float(np.std([r['psnr'] for r in fold_results])),
        'category_metrics': {
            'A': {
                'ssim_mean': float(np.mean([r['ssim_A'] for r in fold_results])),
                'psnr_mean': float(np.mean([r['psnr_A'] for r in fold_results])),
            },
            'B': {
                'ssim_mean': float(np.mean([r['ssim_B'] for r in fold_results])),
                'psnr_mean': float(np.mean([r['psnr_B'] for r in fold_results])),
            },
            'C': {
                'ssim_mean': float(np.mean([r['ssim_C'] for r in fold_results])),
                'psnr_mean': float(np.mean([r['psnr_C'] for r in fold_results])),
            },
        },
        'worst_case': {
            'ssim_C_worst20_mean': float(np.mean([r['ssim_C_worst20'] for r in fold_results])),
            'ssim_C_worst20_min': float(min([r['ssim_C_worst20'] for r in fold_results])),
            'psnr_C_worst20_mean': float(np.mean([r['psnr_C_worst20'] for r in fold_results])),
        },
        'fold_results': fold_results,
        'best_fold': best_fold + 1,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Cross-Validation Complete!")
    print(f"{'='*60}")
    print(f"Overall SSIM: {cv_results['ssim_mean']:.4f} ± {cv_results['ssim_std']:.4f}")
    print(f"Overall PSNR: {cv_results['psnr_mean']:.2f} ± {cv_results['psnr_std']:.2f}")
    print(f"\nCategory-wise SSIM:")
    print(f"  A: {cv_results['category_metrics']['A']['ssim_mean']:.4f}")
    print(f"  B: {cv_results['category_metrics']['B']['ssim_mean']:.4f}")
    print(f"  C: {cv_results['category_metrics']['C']['ssim_mean']:.4f}")
    print(f"\n⚠️  WORST-CASE (C bottom 20%):")
    print(f"  SSIM mean: {cv_results['worst_case']['ssim_C_worst20_mean']:.4f}")
    print(f"  SSIM min:  {cv_results['worst_case']['ssim_C_worst20_min']:.4f}  ← LB刺されポイント")
    print(f"\nBest fold: {best_fold + 1} (SSIM: {best_overall_ssim:.4f})")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    # Save final metrics
    final_output = {
        "experiment_id": os.environ.get("EXPERIMENT_ID", "kaggle_run"),
        "timestamp": datetime.now().isoformat(),
        "commit_sha": os.environ.get("COMMIT_SHA", "unknown"),
        "branch": os.environ.get("BRANCH_NAME", "unknown"),
        "cv_results": cv_results,
        "metrics": {
            "ssim": cv_results['ssim_mean'],
            "psnr": cv_results['psnr_mean'],
            "ssim_std": cv_results['ssim_std'],
            "psnr_std": cv_results['psnr_std'],
        },
        "training_time_seconds": int(training_time),
        "config": {
            "n_folds": n_folds,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "image_size": config.image_size,
        }
    }
    
    with open(config.output_dir / "metrics.json", "w") as f:
        json.dump(final_output, f, indent=2)
    
    with open(config.output_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    
    return cv_results


def validate_worst_val(model, df, worst_val_idx, data_dir, device, image_size):
    """Validate on the fixed worst_val set (dark_ratio top 20% of C)."""
    model.eval()
    
    worst_val_df = df.loc[worst_val_idx]
    dataset = OrganoidDataset(worst_val_df, data_dir, image_size, is_test=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    
    ssim_scores = []
    psnr_scores = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            outputs = torch.clamp(model(inputs), 0, 1)
            
            pred_np = outputs.cpu().numpy()
            target_np = targets.cpu().numpy()
            
            for i in range(pred_np.shape[0]):
                p = np.clip(pred_np[i, 0], 0, 1)
                t = np.clip(target_np[i, 0], 0, 1)
                ssim_scores.append(ssim(t, p, data_range=1.0))
                psnr_scores.append(psnr(t, p, data_range=1.0))
    
    return {
        'ssim_worst_val_mean': float(np.mean(ssim_scores)),
        'ssim_worst_val_min': float(np.min(ssim_scores)),
        'ssim_worst_val_std': float(np.std(ssim_scores)),
        'psnr_worst_val_mean': float(np.mean(psnr_scores)),
    }


def train_worst_case_cv(config, n_folds=5):
    """
    Train using Worst-Case Controlled CV.
    
    Key differences from train_kfold:
    1. worst_val is fixed across all folds (dark_ratio top 20% of C)
    2. c_hard_train is fixed in train (60% of C_hard)
    3. KPI is ssim_worst_val_min (not mean SSIM)
    """
    start_time = time.time()
    set_seed(config.seed)
    
    print(f"{'='*60}")
    print(f"Worst-Case Controlled {n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"Epochs per fold: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    
    # Load full dataframe
    df = pd.read_csv(config.train_csv)
    print(f"Total samples: {len(df)}")
    print(f"Category distribution: {df['category'].value_counts().to_dict()}")
    
    # Create worst-case splits
    splits = create_worst_case_splits(df, config.data_dir)
    df = splits['df']  # df with dark_ratio
    worst_val_idx = splits['worst_val_idx']
    c_hard_train_idx = splits['c_hard_train_idx']
    trainable_idx = splits['trainable_idx']
    
    # Create sub-dataframe for K-Fold (excludes worst_val and c_hard_train)
    df_trainable = df.loc[trainable_idx].reset_index(drop=True)
    
    # Stratified K-Fold on trainable samples (by category)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    
    fold_results = []
    worst_val_results = []
    best_overall_ssim = 0
    best_fold = -1
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_trainable, df_trainable['category'])):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Convert back to original indices
        train_original_idx = df_trainable.iloc[train_idx].index.tolist()
        val_original_idx = df_trainable.iloc[val_idx].index.tolist()
        
        # Add c_hard_train to train set (fixed)
        train_all_idx = train_original_idx + c_hard_train_idx
        
        print(f"Train: {len(train_all_idx)} (incl. {len(c_hard_train_idx)} fixed C_hard)")
        print(f"Val: {len(val_original_idx)}")
        print(f"Worst-Val (fixed): {len(worst_val_idx)}")
        
        # Create datasets
        train_df = df.loc[train_all_idx]
        val_df = df.loc[val_original_idx]
        
        train_dataset = OrganoidDataset(train_df, config.data_dir, config.image_size, is_test=False)
        val_dataset = OrganoidDataset(val_df, config.data_dir, config.image_size, is_test=False)
        
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
        
        # Create fresh model for each fold
        model = create_model(config)
        
        criterion = CombinedLoss(config.l1_weight, config.ssim_weight)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
        
        # Training loop for this fold
        best_fold_worst_val = 0
        
        for epoch in range(config.epochs):
            print(f"\nFold {fold+1} - Epoch {epoch + 1}/{config.epochs}")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
            val_metrics = validate_with_categories(model, val_loader, criterion, config.device)
            
            # Evaluate worst_val (the key metric)
            worst_val_metrics = validate_worst_val(
                model, df, worst_val_idx, config.data_dir, config.device, config.image_size
            )
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"⚠️  Worst-Val SSIM: {worst_val_metrics['ssim_worst_val_mean']:.4f} "
                  f"(min: {worst_val_metrics['ssim_worst_val_min']:.4f})")
            
            # Save best model based on worst_val performance
            if worst_val_metrics['ssim_worst_val_mean'] > best_fold_worst_val:
                best_fold_worst_val = worst_val_metrics['ssim_worst_val_mean']
                torch.save(model.state_dict(), config.output_dir / f"best_model_fold{fold}.pth")
        
        # Store fold results
        final_val_metrics = validate_with_categories(model, val_loader, criterion, config.device)
        final_worst_val = validate_worst_val(
            model, df, worst_val_idx, config.data_dir, config.device, config.image_size
        )
        
        fold_results.append({
            'fold': fold + 1,
            'ssim': final_val_metrics['ssim'],
            'psnr': final_val_metrics['psnr'],
            'ssim_A': final_val_metrics['ssim_A'],
            'ssim_B': final_val_metrics['ssim_B'],
            'ssim_C': final_val_metrics['ssim_C'],
        })
        
        worst_val_results.append({
            'fold': fold + 1,
            'ssim_worst_val_mean': final_worst_val['ssim_worst_val_mean'],
            'ssim_worst_val_min': final_worst_val['ssim_worst_val_min'],
            'ssim_worst_val_std': final_worst_val['ssim_worst_val_std'],
        })
        
        if final_worst_val['ssim_worst_val_mean'] > best_overall_ssim:
            best_overall_ssim = final_worst_val['ssim_worst_val_mean']
            best_fold = fold
            torch.save(model.state_dict(), config.output_dir / "best_model.pth")
    
    training_time = time.time() - start_time
    
    # Aggregate results
    cv_results = {
        'n_folds': n_folds,
        'cv_mode': 'worst_case_controlled',
        'ssim_mean': float(np.mean([r['ssim'] for r in fold_results])),
        'ssim_std': float(np.std([r['ssim'] for r in fold_results])),
        'psnr_mean': float(np.mean([r['psnr'] for r in fold_results])),
        'psnr_std': float(np.std([r['psnr'] for r in fold_results])),
        'category_metrics': {
            'A': {'ssim_mean': float(np.mean([r['ssim_A'] for r in fold_results]))},
            'B': {'ssim_mean': float(np.mean([r['ssim_B'] for r in fold_results]))},
            'C': {'ssim_mean': float(np.mean([r['ssim_C'] for r in fold_results]))},
        },
        'worst_val': {
            'n_samples': len(worst_val_idx),
            'ssim_mean': float(np.mean([r['ssim_worst_val_mean'] for r in worst_val_results])),
            'ssim_min': float(min([r['ssim_worst_val_min'] for r in worst_val_results])),
            'ssim_std_across_folds': float(np.std([r['ssim_worst_val_mean'] for r in worst_val_results])),
        },
        'fold_results': fold_results,
        'worst_val_results': worst_val_results,
        'best_fold': best_fold + 1,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Worst-Case Controlled CV Complete!")
    print(f"{'='*60}")
    print(f"Overall SSIM: {cv_results['ssim_mean']:.4f} ± {cv_results['ssim_std']:.4f}")
    print(f"\nCategory-wise SSIM:")
    print(f"  A: {cv_results['category_metrics']['A']['ssim_mean']:.4f}")
    print(f"  B: {cv_results['category_metrics']['B']['ssim_mean']:.4f}")
    print(f"  C: {cv_results['category_metrics']['C']['ssim_mean']:.4f}")
    print(f"\n🎯 WORST-VAL (Fixed, {len(worst_val_idx)} samples):")
    print(f"  SSIM mean: {cv_results['worst_val']['ssim_mean']:.4f}")
    print(f"  SSIM min:  {cv_results['worst_val']['ssim_min']:.4f}  ← NEW KPI!")
    print(f"  Std across folds: {cv_results['worst_val']['ssim_std_across_folds']:.4f}")
    print(f"\nBest fold: {best_fold + 1}")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    # Save final metrics
    final_output = {
        "experiment_id": os.environ.get("EXPERIMENT_ID", "kaggle_run"),
        "timestamp": datetime.now().isoformat(),
        "commit_sha": os.environ.get("COMMIT_SHA", "unknown"),
        "branch": os.environ.get("BRANCH_NAME", "unknown"),
        "cv_mode": "worst_case_controlled",
        "cv_results": cv_results,
        "metrics": {
            "ssim": cv_results['ssim_mean'],
            "psnr": cv_results['psnr_mean'],
            "ssim_std": cv_results['ssim_std'],
            "psnr_std": cv_results['psnr_std'],
            "ssim_worst_val_min": cv_results['worst_val']['ssim_min'],
        },
        "training_time_seconds": int(training_time),
        "config": {
            "n_folds": n_folds,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "image_size": config.image_size,
        }
    }
    
    with open(config.output_dir / "metrics.json", "w") as f:
        json.dump(final_output, f, indent=2)
    
    with open(config.output_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    
    return cv_results


def train_worst_case_cv_v5(config, n_folds=5):
    """
    v5: Train with worst_val integrated into training.
    
    Key changes:
    - worst_train (top 10% of C): Train with Loss×3
    - worst_eval (10-20% of C): Evaluation only
    - c_hard_train: Train with Loss×2
    """
    start_time = time.time()
    set_seed(config.seed)
    
    print(f"{'='*60}")
    print(f"v5: Worst-Case Integrated {n_folds}-Fold CV")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"Epochs per fold: {config.epochs}")
    
    # Load full dataframe
    df = pd.read_csv(config.train_csv)
    print(f"Total samples: {len(df)}")
    
    # Create v5 splits
    splits = create_worst_case_splits_v5(df, config.data_dir)
    df = splits['df']
    worst_train_idx = splits['worst_train_idx']
    worst_eval_idx = splits['worst_eval_idx']
    c_hard_train_idx = splits['c_hard_train_idx']
    trainable_idx = splits['trainable_idx']
    sample_weights = splits['sample_weights']
    
    # Create sub-dataframe for K-Fold
    df_trainable = df.loc[trainable_idx].reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    
    fold_results = []
    worst_eval_results = []
    best_overall_ssim = 0
    best_fold = -1
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_trainable, df_trainable['category'])):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Get original indices
        train_original_idx = df_trainable.iloc[train_idx].index.tolist()
        val_original_idx = df_trainable.iloc[val_idx].index.tolist()
        
        # Combine train indices: trainable_fold + worst_train + c_hard_train
        train_all_idx = train_original_idx + worst_train_idx + c_hard_train_idx
        
        print(f"Train: {len(train_all_idx)} (incl. {len(worst_train_idx)} worst_train×3, {len(c_hard_train_idx)} c_hard×2)")
        print(f"Val: {len(val_original_idx)}")
        print(f"Worst-Eval (fixed): {len(worst_eval_idx)}")
        
        # Create datasets with sample weights
        train_df = df.loc[train_all_idx]
        val_df = df.loc[val_original_idx]
        
        train_dataset = OrganoidDataset(
            train_df, config.data_dir, config.image_size, 
            is_test=False, sample_weights=sample_weights
        )
        val_dataset = OrganoidDataset(
            val_df, config.data_dir, config.image_size, is_test=False
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
        
        # Create model
        model = create_model(config)
        criterion = CombinedLoss(config.l1_weight, config.ssim_weight)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
        
        best_fold_worst_eval = 0
        
        for epoch in range(config.epochs):
            print(f"\nFold {fold+1} - Epoch {epoch + 1}/{config.epochs}")
            
            # Use weighted training
            train_loss = train_epoch_weighted(model, train_loader, criterion, optimizer, config.device)
            val_metrics = validate_with_categories(model, val_loader, criterion, config.device)
            
            # Evaluate on worst_eval (fixed set)
            worst_eval_metrics = validate_worst_val(
                model, df, worst_eval_idx, config.data_dir, config.device, config.image_size
            )
            
            scheduler.step()
            
            print(f"Train Loss (weighted): {train_loss:.4f}")
            print(f"Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"🎯 Worst-Eval SSIM: {worst_eval_metrics['ssim_worst_val_mean']:.4f} "
                  f"(min: {worst_eval_metrics['ssim_worst_val_min']:.4f})")
            
            if worst_eval_metrics['ssim_worst_val_mean'] > best_fold_worst_eval:
                best_fold_worst_eval = worst_eval_metrics['ssim_worst_val_mean']
                torch.save(model.state_dict(), config.output_dir / f"best_model_fold{fold}.pth")
        
        # Store results
        final_val = validate_with_categories(model, val_loader, criterion, config.device)
        final_worst_eval = validate_worst_val(
            model, df, worst_eval_idx, config.data_dir, config.device, config.image_size
        )
        
        fold_results.append({
            'fold': fold + 1,
            'ssim': final_val['ssim'],
            'psnr': final_val['psnr'],
            'ssim_A': final_val['ssim_A'],
            'ssim_B': final_val['ssim_B'],
            'ssim_C': final_val['ssim_C'],
        })
        
        worst_eval_results.append({
            'fold': fold + 1,
            'ssim_worst_eval_mean': final_worst_eval['ssim_worst_val_mean'],
            'ssim_worst_eval_min': final_worst_eval['ssim_worst_val_min'],
            'ssim_worst_eval_std': final_worst_eval['ssim_worst_val_std'],
        })
        
        if final_worst_eval['ssim_worst_val_mean'] > best_overall_ssim:
            best_overall_ssim = final_worst_eval['ssim_worst_val_mean']
            best_fold = fold
            torch.save(model.state_dict(), config.output_dir / "best_model.pth")
    
    training_time = time.time() - start_time
    
    # Aggregate results
    cv_results = {
        'n_folds': n_folds,
        'cv_mode': 'worst_case_v5',
        'ssim_mean': float(np.mean([r['ssim'] for r in fold_results])),
        'ssim_std': float(np.std([r['ssim'] for r in fold_results])),
        'psnr_mean': float(np.mean([r['psnr'] for r in fold_results])),
        'category_metrics': {
            'A': {'ssim_mean': float(np.mean([r['ssim_A'] for r in fold_results]))},
            'B': {'ssim_mean': float(np.mean([r['ssim_B'] for r in fold_results]))},
            'C': {'ssim_mean': float(np.mean([r['ssim_C'] for r in fold_results]))},
        },
        'worst_eval': {
            'n_samples': len(worst_eval_idx),
            'ssim_mean': float(np.mean([r['ssim_worst_eval_mean'] for r in worst_eval_results])),
            'ssim_min': float(min([r['ssim_worst_eval_min'] for r in worst_eval_results])),
            'ssim_std_across_folds': float(np.std([r['ssim_worst_eval_mean'] for r in worst_eval_results])),
        },
        'fold_results': fold_results,
        'worst_eval_results': worst_eval_results,
        'best_fold': best_fold + 1,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"v5 Worst-Case Integrated CV Complete!")
    print(f"{'='*60}")
    print(f"Overall SSIM: {cv_results['ssim_mean']:.4f} ± {cv_results['ssim_std']:.4f}")
    print(f"\nCategory-wise SSIM:")
    print(f"  A: {cv_results['category_metrics']['A']['ssim_mean']:.4f}")
    print(f"  B: {cv_results['category_metrics']['B']['ssim_mean']:.4f}")
    print(f"  C: {cv_results['category_metrics']['C']['ssim_mean']:.4f}")
    print(f"\n🎯 WORST-EVAL (Fixed, {len(worst_eval_idx)} samples):")
    print(f"  SSIM mean: {cv_results['worst_eval']['ssim_mean']:.4f}")
    print(f"  SSIM min:  {cv_results['worst_eval']['ssim_min']:.4f}  ← v5 KPI")
    print(f"  Std across folds: {cv_results['worst_eval']['ssim_std_across_folds']:.4f}")
    print(f"\nBest fold: {best_fold + 1}")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    # Save metrics
    final_output = {
        "experiment_id": os.environ.get("EXPERIMENT_ID", "kaggle_run"),
        "timestamp": datetime.now().isoformat(),
        "commit_sha": os.environ.get("COMMIT_SHA", "unknown"),
        "branch": os.environ.get("BRANCH_NAME", "unknown"),
        "cv_mode": "worst_case_v5",
        "cv_results": cv_results,
        "metrics": {
            "ssim": cv_results['ssim_mean'],
            "psnr": cv_results['psnr_mean'],
            "ssim_std": cv_results['ssim_std'],
            "ssim_worst_eval_min": cv_results['worst_eval']['ssim_min'],
        },
        "training_time_seconds": int(training_time),
        "config": {
            "n_folds": n_folds,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "image_size": config.image_size,
        }
    }
    
    with open(config.output_dir / "metrics.json", "w") as f:
        json.dump(final_output, f, indent=2)
    
    with open(config.output_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    
    return cv_results


# ==============================================================================
# Inference and Submission
# ==============================================================================

def predict_and_submit(config, model_path=None):
    """
    Run inference on test set and create submission.
    Saves predicted images to output_dir/submission/
    """
    print(f"\n{'='*60}")
    print("Running Inference on Test Set")
    print(f"{'='*60}")
    
    # Load model
    model = create_model(config)
    
    if model_path is None:
        model_path = config.output_dir / "best_model.pth"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Load test data
    test_df = pd.read_csv(config.test_csv)
    print(f"Test samples: {len(test_df)}")
    
    test_dataset = OrganoidDataset(
        test_df, config.data_dir, config.image_size, is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Create submission directory
    submission_dir = config.output_dir / "submission"
    submission_dir.mkdir(exist_ok=True)
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            inputs = batch["input"].to(config.device)
            ids = batch["id"]
            
            outputs = torch.clamp(model(inputs), 0, 1)
            
            # Save each prediction
            for i, sample_id in enumerate(ids):
                pred = outputs[i, 0].cpu().numpy()
                pred_uint8 = (pred * 255).astype(np.uint8)
                
                # Save as PNG
                img = Image.fromarray(pred_uint8)
                img.save(submission_dir / f"{sample_id}.png")
    
    print(f"\nSubmission saved to: {submission_dir}")
    print(f"Total predictions: {len(list(submission_dir.glob('*.png')))}")
    
    return submission_dir


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
    
    # CV mode selection
    cv_mode = os.environ.get("CV_MODE", "worst_case_v5")  # Default: v5
    n_folds = int(os.environ.get("N_FOLDS", "5"))
    run_inference = os.environ.get("RUN_INFERENCE", "1") == "1"  # Default: run inference
    
    print(f"CV Mode: {cv_mode}")
    print(f"Run Inference: {run_inference}")
    
    # Training
    if cv_mode == "worst_case_v5":
        train_worst_case_cv_v5(config, n_folds=n_folds)
    elif cv_mode == "worst_case":
        train_worst_case_cv(config, n_folds=n_folds)
    elif cv_mode == "kfold" and n_folds > 1:
        train_kfold(config, n_folds=n_folds)
    else:
        train(config)
    
    # Inference (for LB submission)
    if run_inference:
        predict_and_submit(config)

