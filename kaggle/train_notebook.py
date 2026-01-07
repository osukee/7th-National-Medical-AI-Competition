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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
# Category C Sub-Clustering
# ==============================================================================

def cluster_category_c(df, data_dir, n_clusters=3):
    """
    Cluster Category C samples into difficulty sub-groups.
    Uses image brightness, contrast, and dark pixel ratio as features.
    Returns df with 'difficulty' column (A, B, C_easy, C_medium, C_hard)
    """
    print("Clustering Category C into difficulty sub-groups...")
    
    # Initialize difficulty column
    df = df.copy()
    df['difficulty'] = df['category']  # Default: A, B stay as-is
    
    df_c = df[df['category'] == 'C']
    if len(df_c) == 0:
        return df
    
    # Extract features from C samples
    features = []
    valid_indices = []
    
    for idx, row in df_c.iterrows():
        try:
            input_path = Path(data_dir) / row['input_path']
            img = Image.open(input_path).convert('L')
            arr = np.array(img)
            
            brightness = arr.mean()
            contrast = arr.std()
            dark_ratio = (arr < 50).sum() / arr.size
            
            features.append([brightness, contrast, dark_ratio])
            valid_indices.append(idx)
        except Exception:
            continue
    
    if len(features) < n_clusters:
        print(f"Not enough valid C samples for clustering: {len(features)}")
        return df
    
    features = np.array(features)
    
    # Standardize and cluster
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Determine difficulty by dark_ratio (higher = harder)
    cluster_dark_ratios = []
    for c in range(n_clusters):
        mask = clusters == c
        mean_dark = features[mask, 2].mean()  # dark_ratio is index 2
        cluster_dark_ratios.append((c, mean_dark))
    
    # Sort by dark ratio (ascending = easy to hard)
    cluster_dark_ratios.sort(key=lambda x: x[1])
    
    labels = ['C_easy', 'C_medium', 'C_hard'] if n_clusters == 3 else ['C_easy', 'C_hard']
    difficulty_map = {c: labels[i] for i, (c, _) in enumerate(cluster_dark_ratios)}
    
    # Assign difficulty labels
    for i, idx in enumerate(valid_indices):
        df.loc[idx, 'difficulty'] = difficulty_map[clusters[i]]
    
    # Print distribution
    print(f"Difficulty distribution: {df['difficulty'].value_counts().to_dict()}")
    
    return df


# ==============================================================================
# Dataset
# ==============================================================================

class OrganoidDataset(Dataset):
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
        
        # Include category for stratified evaluation
        category = row.get("category", "unknown")
        
        return {"id": row["id"], "input": input_tensor, "target": target_tensor, "category": category}


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
    
    # Use K-Fold CV if N_FOLDS is set, otherwise default to 5-fold
    n_folds = int(os.environ.get("N_FOLDS", "5"))  # Default: 5-fold CV
    
    if n_folds > 1:
        train_kfold(config, n_folds=n_folds)
    else:
        train(config)

