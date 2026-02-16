"""
Medical AI Competition Training Script for Kaggle
This script is designed to run on Kaggle's GPU environment.
"""

# Install segmentation-models-pytorch if not available
import subprocess
import sys
try:
    import segmentation_models_pytorch
except ImportError:
    print("Installing segmentation-models-pytorch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"])
    import segmentation_models_pytorch  # Re-import after install
    print("SMP installed successfully!")

# Install albumentations if not available (exp_022)
try:
    import albumentations as A
except ImportError:
    print("Installing albumentations...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "albumentations"])
    import albumentations as A
    print("Albumentations installed successfully!")

# Install trackio for experiment tracking
try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    print("Installing trackio...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "trackio"])
    try:
        import trackio
        TRACKIO_AVAILABLE = True
        print("Trackio installed successfully!")
    except ImportError:
        TRACKIO_AVAILABLE = False
        print("Trackio not available, skipping experiment tracking.")

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
    image_size = 512  # Back to 512 baseline
    in_channels = 1
    out_channels = 1
    
    # Training
    epochs = 20  # auto-improved
    batch_size = 4  # auto-improved
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_workers = 2
    
    # Loss weights (for EdgeAwareLoss - backward compat)
    l1_weight = 1.0
    ssim_weight = 1.0  # exp_019: Back to optimal value from exp_017e
    mask_outside_weight = 0.2  # Loss weight for mask-outside region (0 = ignore, 1 = full)
    edge_weight = 0.1  # Weight for edge loss (0.05-0.2 recommended)
    
    # OptimizedLoss weights (for grid search experiments)
    # Loss = α*L1 + β*(1-SSIM) + γ*GradLoss + δ*TV
    grad_weight = 0.5     # γ: gradient/edge preservation (0.2-1.0)
    tv_weight = 1e-4      # δ: total variation for noise (1e-4 to 1e-3)
    lambda_edge = 2.0     # Edge weighting multiplier for EdgeWeightedLoss
    
    # Loss function selection
    # Options: "combined", "masked", "edge_aware", "optimized", "edge_weighted"
    # exp_021: Fixed EdgeWeightedLoss with SSIM + Grad + edge-weighted L1
    loss_type = "optimized"  # exp_031: Codex rec - edge_weighted was fragile in exp_020
    
    # Model - exp_016: Upgrade to efficientnet-b4 for better feature extraction
    encoder = "efficientnet-b5"  # auto-improved
    encoder_weights = "imagenet"
    gradient_checkpointing = True  # auto-improved
    
    # Architecture selection
    # Options: "unet", "unetplusplus" (U-Net++)
    # exp_019: Test U-Net++ for better multi-scale feature fusion
    architecture = "unetplusplus"
    
    # exp_013: Distribution analysis settings
    analyze_distribution = True  # Enable distribution analysis on validation
    
    # exp_016: Mean Matching DISABLED - exp_015 showed it hurts LB
    # Previous exp_014/015: mean_matching_enabled=True, delta=16.1
    # Result: LB 0.407 (same as 0.410 baseline → no improvement)
    mean_matching_enabled = False  # Disabled for exp_016
    mean_matching_delta = 0.0      # Not used
    
    # exp_018: Test Time Augmentation (TTA)
    # Predict with original + horizontal flip + vertical flip + both, average results
    tta_enabled = True  # Enable TTA for inference
    tta_mode = "dihedral8"  # auto-improved
    tta_aggregate = "median"  # "median" or "mean"
    
    # exp_022/023: Data Augmentation (training only)
    # exp_022 failed because brightness/contrast broke input-target correspondence
    # exp_023: Redesigned to use geometric-only transforms
    augmentation_enabled = True  # Enable data augmentation during training
    augmentation_strength = 0.7  # auto-improved
    augmentation_mode = 'geometric'  # 'geometric' (safe) or 'intensity' (deprecated)
    
    # exp_025: Fold Selection Ensemble
    # Select top N folds by SSIM (reject weak folds to reduce noise)
    # rank-based weights instead of softmax (preserves differentiation)
    n_folds_ensemble = 3              # exp_031: Codex rec - weak folds add noise (exp_025)
    fold_rank_weights = [1.0, 0.7, 0.4]  # exp_031: Proven top-3 weights
    
    # Post-processing options (Phase A quick wins)
    median_filter_size = 0    # 0=disabled, 3=3x3 median (salt-pepper removal)
    unsharp_strength = 0.0    # 0=disabled, 0.5=recommended (edge enhancement)
    unsharp_radius = 1        # Radius for unsharp mask
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seed
    seed = 42
    
    # Experiment tracking
    tracking_enabled = True
    tracking_project = "medical-ai-7th"
    
    # exp_027: VirtualStaining Transfer Learning
    transfer_learning_enabled = False  # exp_030: Disabled (VirtualStaining was rule violation)
    pretrained_encoder_path = "/kaggle/input/virtualstaining-pretrained/best_model.pth"
    encoder_lr = 1e-5   # Low LR for encoder (fine-tune)
    decoder_lr = 1e-4   # Normal LR for decoder
    freeze_encoder = False  # Do NOT freeze (low LR is better)
    
    # exp_028: CLAHE preprocessing (match VirtualStaining)
    clahe_enabled = True
    clahe_clip_limit = 2.0
    clahe_tile_size = (8, 8)
    
    # exp_030: Pseudo-Labeling
    # exp_030: Pseudo-Labeling Phase 2 (disabled for Phase 2 stability check)
    pseudo_label_enabled = False  # Disabled until debugged
    pseudo_label_epochs = 10       # Phase 2 fine-tuning epochs
    pseudo_label_weight = 0.5      # Loss weight for pseudo-labeled samples (vs 1.0 for real)
    pseudo_label_lr_factor = 0.3   # LR = learning_rate * factor for Phase 2

# ==============================================================================
# Excluded Samples (all-zero target images)
# ==============================================================================
# These 4 training samples have organoids not properly visible in transmission
# images, resulting in all-zero target fluorescence images.
# They should be excluded from training as they add noise to the loss.
EXCLUDED_SAMPLE_IDS = {
    'train_00099',
    'train_00603', 
    'train_00802',
    'train_00863'
}


def filter_excluded_samples(df):
    """Remove samples with all-zero targets from dataframe.
    
    IMPORTANT: Returns df with reset_index to avoid KeyError in downstream 
    functions that use df.index for splitting (e.g., create_worst_case_splits).
    """
    before_count = len(df)
    df_filtered = df[~df['id'].isin(EXCLUDED_SAMPLE_IDS)].reset_index(drop=True)
    after_count = len(df_filtered)
    excluded = before_count - after_count
    if excluded > 0:
        print(f"Excluded {excluded} samples with all-zero targets: {EXCLUDED_SAMPLE_IDS}")
    return df_filtered


def calculate_lb_score(ssim_val, psnr_val):
    """
    Calculate LB score using official Kaggle formula.
    
    Score = (SSIM + PSNR_norm) / 2
    
    Where:
    - SSIM: 0-1 range
    - PSNR_norm = clip((PSNR - 15) / 20, 0, 1)
      - 15 dB → 0.0
      - 35 dB → 1.0
    """
    psnr_norm = np.clip((psnr_val - 15) / 20, 0, 1)
    return (ssim_val + psnr_norm) / 2

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
# Category C Clustering (for difficulty-based stratification)
# ==============================================================================

def cluster_category_c(df, data_dir, n_clusters=3):
    """
    Cluster Category C samples based on image features.
    Returns df with added 'difficulty' column.
    
    Note: This is a simplified version that avoids sklearn KMeans dependency issues.
    Uses dark_ratio quantiles for clustering instead.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    print("="*60)
    print("Category C Sub-Clustering Analysis")
    print("="*60)
    
    # Filter to Category C only
    df_c = df[df['category'] == 'C'].copy()
    print(f"Category C samples: {len(df_c)}")
    
    if len(df_c) == 0:
        df['difficulty'] = df['category']
        return df
    
    # Extract features
    features = []
    valid_indices = []
    
    for idx in df_c.index:
        row = df_c.loc[idx]
        try:
            input_path = Path(data_dir) / row['input_path']
            img = Image.open(input_path).convert('L')
            arr = np.array(img)
            
            brightness = float(arr.mean())
            contrast = float(arr.std())
            dark_ratio = float((arr < 50).sum() / arr.size)
            
            features.append([brightness, contrast, dark_ratio])
            valid_indices.append(idx)
        except Exception:
            continue
    
    if len(features) == 0:
        df['difficulty'] = df['category']
        return df
    
    features = np.array(features)
    print(f"Extracted features for {len(features)} samples")
    
    # Handle edge case when n_clusters > number of samples
    actual_n_clusters = min(n_clusters, len(features))
    if actual_n_clusters < n_clusters:
        print(f"Warning: Reducing n_clusters from {n_clusters} to {actual_n_clusters}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters to determine difficulty
    cluster_stats = []
    for c in range(actual_n_clusters):
        mask = clusters == c
        cluster_features = features[mask]
        if len(cluster_features) > 0:
            cluster_stats.append({
                'cluster': c,
                'count': int(mask.sum()),
                'mean_dark_ratio': float(cluster_features[:, 2].mean()),
            })
    
    # Sort by difficulty (higher dark_ratio = harder)
    cluster_stats.sort(key=lambda x: x['mean_dark_ratio'])
    
    # Map clusters to difficulty labels
    difficulty_labels = ['C_easy', 'C_medium', 'C_hard'][:actual_n_clusters]
    difficulty_map = {}
    for i, stat in enumerate(cluster_stats):
        difficulty_map[stat['cluster']] = difficulty_labels[i]
    
    # Assign difficulty labels to dataframe
    df['difficulty'] = df['category']  # Default: use category as difficulty
    for i, idx in enumerate(valid_indices):
        cluster_id = clusters[i]
        df.loc[idx, 'difficulty'] = difficulty_map[cluster_id]
    
    print(f"Difficulty distribution: {df['difficulty'].value_counts().to_dict()}")
    
    return df


# ==============================================================================
# Dataset
# ==============================================================================

def get_training_augmentation(strength=0.5, mode='geometric'):
    """
    Create augmentation pipeline for Image-to-Image training.
    
    exp_023: Redesigned augmentation strategy.
    
    Key insight from exp_022 failure:
    - Brightness/contrast changes BREAK input-target correspondence
    - GaussNoise degrades the signal the model needs to learn from
    - Only GEOMETRIC transforms are safe for Image-to-Image
    
    Modes:
    - 'geometric': Safe transforms for Image-to-Image (flip, rotate, elastic)
    - 'intensity': DEPRECATED - breaks correspondence (kept for comparison)
    
    Args:
        strength: Probability for augmentations (0.3=weak, 0.5=medium, 0.7=strong)
        mode: 'geometric' (recommended) or 'intensity' (deprecated)
    
    Returns:
        albumentations.Compose pipeline
    """
    if mode == 'geometric':
        # exp_023: Geometric-only augmentation (safe for Image-to-Image)
        return A.Compose([
            # Flip transforms (absolutely safe - no interpolation)
            A.HorizontalFlip(p=strength),
            A.VerticalFlip(p=strength),
            
            # Rotation with 90-degree increments (no interpolation artifacts)
            A.RandomRotate90(p=strength),
            
            # Small affine transforms (interpolation but structure-preserving)
            A.ShiftScaleRotate(
                shift_limit=0.03,      # Reduced from 0.05
                scale_limit=0.05,      # Reduced from 0.1
                rotate_limit=10,       # Reduced from 15
                border_mode=0,         # Constant padding (black)
                p=strength * 0.5       # Lower probability
            ),
            
            # Elastic transform - simulates cell deformation (key for organoid data)
            A.ElasticTransform(
                alpha=50,              # Deformation intensity
                sigma=5,               # Smoothness of deformation
                border_mode=0,
                p=strength * 0.3       # Use sparingly
            ),
            
            # exp_030: GridDistortion - simulates organoid boundary diversity
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                border_mode=0,
                p=strength * 0.2
            ),
        ], additional_targets={'target': 'image', 'mask': 'mask'})
    
    else:
        # DEPRECATED: Original exp_022 approach (kept for A/B comparison)
        # WARNING: This breaks input-target correspondence!
        return A.Compose([
            A.HorizontalFlip(p=strength),
            A.VerticalFlip(p=strength),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.1, 
                rotate_limit=15, 
                border_mode=0,
                p=strength
            ),
            # These HARM the model:
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=strength
            ),
            A.GaussNoise(var_limit=(0.001, 0.01), p=strength * 0.5),
        ], additional_targets={'target': 'image', 'mask': 'mask'})


class OrganoidDataset(Dataset):
    def __init__(self, csv_path_or_df, data_dir, image_size=512, is_test=False, 
                 indices=None, sample_weights=None, augmentation=None):
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
        
        # exp_022: Augmentation pipeline (training only)
        self.augmentation = augmentation
        
        # exp_028: CLAHE preprocessing (match VirtualStaining)
        self.clahe_enabled = getattr(Config, 'clahe_enabled', False)
        self.clahe_clip_limit = getattr(Config, 'clahe_clip_limit', 2.0)
        self.clahe_tile_size = getattr(Config, 'clahe_tile_size', (8, 8))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        original_idx = self.original_indices[idx] if idx < len(self.original_indices) else idx
        
        # Load input image
        input_path = self.data_dir / row["input_path"]
        input_img = Image.open(input_path).convert("L")
        input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        input_arr = np.array(input_img, dtype=np.uint8)
        
        # exp_028: Apply CLAHE preprocessing
        if self.clahe_enabled:
            import cv2
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size
            )
            input_arr = clahe.apply(input_arr)
        
        input_arr = input_arr.astype(np.float32) / 255.0
        
        if self.is_test:
            input_tensor = torch.from_numpy(input_arr).unsqueeze(0)
            return {"id": row["id"], "input": input_tensor}
        
        # Load target image
        target_path = self.data_dir / row["target_path"]
        target_img = Image.open(target_path).convert("L")
        target_img = target_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        target_arr = np.array(target_img, dtype=np.float32) / 255.0
        
        # Load mask image (for LB-aligned evaluation)
        mask_arr = None
        if "mask_path" in row and pd.notna(row.get("mask_path", None)):
            mask_path = self.data_dir / row["mask_path"]
            if mask_path.exists():
                mask_img = Image.open(mask_path).convert("L")
                mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
                mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
        
        # exp_022: Apply augmentation (training only)
        if self.augmentation is not None:
            # Prepare data for albumentations (expects HWC or HW for grayscale)
            aug_input = {
                'image': input_arr,  # HW grayscale
                'target': target_arr,  # HW grayscale (spatial transforms only)
            }
            if mask_arr is not None:
                aug_input['mask'] = mask_arr
            
            # Apply augmentation
            augmented = self.augmentation(**aug_input)
            input_arr = augmented['image']
            target_arr = augmented['target']
            if mask_arr is not None:
                mask_arr = augmented['mask']
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_arr).unsqueeze(0)
        target_tensor = torch.from_numpy(target_arr).unsqueeze(0)
        
        mask_tensor = None
        if mask_arr is not None:
            mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        
        # Include category for stratified evaluation
        category = row.get("category", "unknown")
        
        # Get sample weight (default 1.0)
        weight = self.sample_weights.get(original_idx, 1.0)
        
        result = {
            "id": row["id"], 
            "input": input_tensor, 
            "target": target_tensor, 
            "category": category,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }
        if mask_tensor is not None:
            result["mask"] = mask_tensor
        
        return result


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
        
        return torch.sigmoid(self.out_conv(d1))  # sigmoid for [0,1] output


def create_model(config):
    """Create model, preferring SMP if available.
    
    Supports:
    - config.architecture = "unet" (default)
    - config.architecture = "unetplusplus" (U-Net++)
    - config.decoder_attention_type = "scse" (optional attention)
    """
    try:
        import segmentation_models_pytorch as smp
        
        # Get decoder attention type (default None for no attention)
        decoder_attention = getattr(config, 'decoder_attention_type', None)
        
        # Get architecture type (default to unet)
        arch = getattr(config, 'architecture', 'unet')
        
        if arch == 'unetplusplus':
            model = smp.UnetPlusPlus(
                encoder_name=config.encoder,
                encoder_weights=config.encoder_weights,
                in_channels=config.in_channels,
                classes=config.out_channels,
                activation='sigmoid',  # Output in [0,1] range
                decoder_attention_type=decoder_attention,
            )
            arch_name = "U-Net++"
        else:
            model = smp.Unet(
                encoder_name=config.encoder,
                encoder_weights=config.encoder_weights,
                in_channels=config.in_channels,
                classes=config.out_channels,
                activation='sigmoid',  # Output in [0,1] range
                decoder_attention_type=decoder_attention,
            )
            arch_name = "U-Net"
        
        attention_str = f" + {decoder_attention} attention" if decoder_attention else ""
        print(f"Using SMP {arch_name} ({config.encoder}{attention_str}) with sigmoid activation")
        
        # exp_027: Load VirtualStaining pretrained encoder weights
        model = _load_transfer_weights(model, config)
        
        # exp_030: Gradient checkpointing to save VRAM with larger encoders
        if getattr(config, 'gradient_checkpointing', False):
            try:
                from torch.utils.checkpoint import checkpoint
                # Enable gradient checkpointing on encoder
                if hasattr(model.encoder, 'set_grad_checkpointing'):
                    model.encoder.set_grad_checkpointing(True)
                    print("Gradient checkpointing enabled (encoder)")
                else:
                    # Fallback: enable on all modules that support it
                    for module in model.encoder.modules():
                        if hasattr(module, 'gradient_checkpointing'):
                            module.gradient_checkpointing = True
                    print("Gradient checkpointing enabled (module-level)")
            except Exception as e:
                print(f"Warning: Could not enable gradient checkpointing: {e}")
        
    except ImportError:
        model = SimpleUNet(config.in_channels, config.out_channels)
        print("Using Simple U-Net (SMP not available) with sigmoid activation")
    
    model = model.to(config.device)
    torch.cuda.empty_cache()  # exp_030: Free fragmented VRAM after model load
    return model


def _load_transfer_weights(model, config):
    """exp_027: Load VirtualStaining encoder weights for transfer learning."""
    import os
    
    transfer_enabled = getattr(config, 'transfer_learning_enabled', False)
    pretrained_path = getattr(config, 'pretrained_encoder_path', None)
    
    if not transfer_enabled or not pretrained_path:
        return model
    
    if not os.path.exists(pretrained_path):
        print(f"⚠️ Transfer learning: pretrained file not found: {pretrained_path}")
        return model
    
    print(f"\n{'='*60}")
    print("exp_027: Loading VirtualStaining Encoder Weights")
    print("="*60)
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extract encoder weights
    encoder_weights = {k: v for k, v in checkpoint.items() if 'encoder' in k.lower()}
    print(f"  Found {len(encoder_weights)} encoder parameters in checkpoint")
    
    # Load encoder weights
    current_state = model.encoder.state_dict()
    loaded_count = 0
    for key, value in encoder_weights.items():
        clean_key = key.replace('encoder.', '')
        if clean_key in current_state and current_state[clean_key].shape == value.shape:
            current_state[clean_key] = value
            loaded_count += 1
    
    model.encoder.load_state_dict(current_state)
    print(f"  Loaded {loaded_count} encoder layers from VirtualStaining")
    
    # Reset BatchNorm statistics
    bn_count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()
            bn_count += 1
    print(f"  Reset {bn_count} BatchNorm layers")
    
    # Freeze encoder if requested (not recommended)
    if getattr(config, 'freeze_encoder', False):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("  ⚠️ Encoder frozen (not recommended)")
    else:
        print("  Encoder will be fine-tuned with low LR")
    
    print("="*60 + "\n")
    return model


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


class MaskedCombinedLoss(nn.Module):
    """
    Combined L1 + SSIM loss with mask-based spatial weighting.
    
    Applies different weights to mask-inside and mask-outside regions
    to improve boundary continuity while focusing on the evaluation region.
    
    Args:
        l1_weight: Weight for L1 loss component
        ssim_weight: Weight for SSIM loss component
        inside_weight: Loss weight for mask inside region (default: 1.0)
        outside_weight: Loss weight for mask outside region (default: 0.2)
    """
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, inside_weight=1.0, outside_weight=0.2):
        super().__init__()
        self.ssim_loss = SSIMLoss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.inside_weight = inside_weight
        self.outside_weight = outside_weight
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)
            mask: Optional mask tensor (B, C, H, W), >0 = inside region
        """
        # SSIM loss (always computed on full image for proper windowing)
        ssim = self.ssim_loss(pred, target)
        
        if mask is not None:
            # Compute spatial weight map: inside=inside_weight, outside=outside_weight
            mask_binary = (mask > 0.5).float()
            weight_map = mask_binary * self.inside_weight + (1 - mask_binary) * self.outside_weight
            
            # Weighted L1 loss
            l1_per_pixel = torch.abs(pred - target)
            weighted_l1 = (l1_per_pixel * weight_map).sum() / weight_map.sum()
            
            return self.l1_weight * weighted_l1 + self.ssim_weight * ssim
        else:
            # Fallback to standard L1 loss
            l1 = nn.functional.l1_loss(pred, target)
            return self.l1_weight * l1 + self.ssim_weight * ssim


class SobelEdgeLoss(nn.Module):
    """
    Edge-aware loss using Sobel operators.
    
    Purpose: Force model to preserve structure instead of producing blurry averages.
    Apply to BOTH prediction and GT, compare their edge maps.
    
    Key insight: Edge loss should have its own mask strategy (mask + outside_base)
    because boundaries extend slightly beyond the mask region.
    """
    def __init__(self, outside_base=0.3):
        super().__init__()
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Register as buffers (move to device with model)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.outside_base = outside_base
    
    def _compute_edges(self, x):
        """Compute edge magnitude from Sobel operators."""
        # Apply Sobel operators
        edge_x = nn.functional.conv2d(x, self.sobel_x, padding=1)
        edge_y = nn.functional.conv2d(x, self.sobel_y, padding=1)
        # Edge magnitude
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)
            mask: Optional mask tensor (B, C, H, W)
        """
        # Compute edge maps for both pred and GT
        edge_pred = self._compute_edges(pred)
        edge_gt = self._compute_edges(target)
        
        if mask is not None:
            # Edge-specific mask: include boundary region (mask + outside_base)
            # This is different from pixel loss mask!
            edge_weight = torch.clamp(mask + self.outside_base, 0, 1)
            
            # Weighted L1 on edge maps
            edge_diff = torch.abs(edge_pred - edge_gt)
            weighted_edge_loss = (edge_diff * edge_weight).sum() / (edge_weight.sum() + 1e-8)
            return weighted_edge_loss
        else:
            # Unmasked edge loss
            return nn.functional.l1_loss(edge_pred, edge_gt)


class EdgeAwareLoss(nn.Module):
    """
    Combined loss with separate pixel and edge components.
    
    Architecture:
    - Pixel loss: L1 + SSIM (masked with inside=1.0, outside=0.2)
    - Edge loss: Sobel L1 (masked with mask + 0.3, different strategy!)
    
    Goal: "Stop averaging, preserve structure"
    
    Args:
        l1_weight: Weight for L1 pixel loss
        ssim_weight: Weight for SSIM loss
        edge_weight: Weight for edge loss (recommend 0.05-0.2)
        mask_outside_weight: Weight for pixel loss outside mask
        edge_outside_base: Base weight for edge loss outside mask
    """
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, edge_weight=0.1, 
                 mask_outside_weight=0.2, edge_outside_base=0.3):
        super().__init__()
        self.pixel_loss = MaskedCombinedLoss(
            l1_weight=l1_weight, 
            ssim_weight=ssim_weight,
            inside_weight=1.0,
            outside_weight=mask_outside_weight
        )
        self.edge_loss = SobelEdgeLoss(outside_base=edge_outside_base)
        self.edge_weight = edge_weight
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)
            mask: Optional mask tensor (B, C, H, W)
        """
        # Pixel-based loss (L1 + SSIM)
        loss_pixel = self.pixel_loss(pred, target, mask)
        
        # Edge-based loss (Sobel)
        loss_edge = self.edge_loss(pred, target, mask)
        
        return loss_pixel + self.edge_weight * loss_edge


class TVLoss(nn.Module):
    """
    Total Variation Loss for noise suppression.
    
    Encourages spatial smoothness by penalizing differences between
    adjacent pixels. Helps reduce salt-and-pepper noise and artifacts.
    
    TV = mean(|I(x+1,y) - I(x,y)|) + mean(|I(x,y+1) - I(x,y)|)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (B, C, H, W)
            mask: Optional mask tensor (B, C, H, W), >0 = evaluate
        """
        # Horizontal differences
        h_diff = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        # Vertical differences
        v_diff = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        if mask is not None:
            # Apply mask to differences
            h_mask = mask[:, :, :, 1:]  # Align with h_diff
            v_mask = mask[:, :, 1:, :]  # Align with v_diff
            
            h_loss = (h_diff * h_mask).sum() / (h_mask.sum() + 1e-8)
            v_loss = (v_diff * v_mask).sum() / (v_mask.sum() + 1e-8)
        else:
            h_loss = h_diff.mean()
            v_loss = v_diff.mean()
        
        return h_loss + v_loss


class OptimizedLoss(nn.Module):
    """
    Optimized combined loss for competition.
    
    Loss = α * L1 + β * (1-SSIM) + γ * GradLoss + δ * TV
    
    Recommended starting values:
    - l1_weight (α) = 1.0
    - ssim_weight (β) = 1.0 → increase to 1.5-2.0 for SSIM focus
    - grad_weight (γ) = 0.5 → edge preservation
    - tv_weight (δ) = 1e-4 → noise suppression (very small)
    
    For grid search experiments:
    - β = [0.5, 1.0, 1.5]
    - γ = [0.2, 0.5, 1.0]
    """
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, grad_weight=0.5, 
                 tv_weight=1e-4, mask_outside_weight=0.2):
        super().__init__()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = SobelEdgeLoss(outside_base=0.3)
        self.tv_loss = TVLoss()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.tv_weight = tv_weight
        self.mask_outside_weight = mask_outside_weight
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)
            mask: Optional mask tensor (B, C, H, W)
        """
        # 1. L1 Loss (with mask weighting)
        if mask is not None:
            mask_binary = (mask > 0.5).float()
            weight_map = mask_binary * 1.0 + (1 - mask_binary) * self.mask_outside_weight
            l1_per_pixel = torch.abs(pred - target)
            loss_l1 = (l1_per_pixel * weight_map).sum() / weight_map.sum()
        else:
            loss_l1 = nn.functional.l1_loss(pred, target)
        
        # 2. SSIM Loss
        loss_ssim = self.ssim_loss(pred, target)
        
        # 3. Gradient Loss (edge preservation)
        loss_grad = self.edge_loss(pred, target, mask)
        
        # 4. TV Loss (noise suppression)
        loss_tv = self.tv_loss(pred, mask)
        
        # Combined
        total_loss = (
            self.l1_weight * loss_l1 +
            self.ssim_weight * loss_ssim +
            self.grad_weight * loss_grad +
            self.tv_weight * loss_tv
        )
        
        return total_loss


class EdgeWeightedLoss(nn.Module):
    """
    Per-pixel edge-weighted loss with SSIM.
    
    Combines:
    1. Edge-weighted L1: L1_pixel * (1 + λ * normalized_edge_map)
    2. SSIM loss: β * (1 - SSIM)
    3. Gradient loss: γ * GradLoss
    
    This forces the model to pay more attention to edge regions,
    which is critical for SSIM improvement.
    
    Args:
        l1_weight: Weight for edge-weighted L1 loss
        ssim_weight: Weight for SSIM loss (critical!)
        grad_weight: Weight for gradient loss
        lambda_edge: Edge weighting multiplier (recommend 2.0)
        mask_outside_weight: Weight for pixels outside mask
    """
    def __init__(self, l1_weight=1.0, ssim_weight=1.0, grad_weight=0.5,
                 lambda_edge=2.0, mask_outside_weight=0.2):
        super().__init__()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = SobelEdgeLoss(outside_base=0.3)
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.lambda_edge = lambda_edge
        self.mask_outside_weight = mask_outside_weight
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def _compute_edge_weight(self, x):
        """Compute normalized edge weight map."""
        edge_x = nn.functional.conv2d(x, self.sobel_x, padding=1)
        edge_y = nn.functional.conv2d(x, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        
        # Normalize to [0, 1] per-sample
        B = edge_mag.shape[0]
        edge_norm = edge_mag.clone()
        for i in range(B):
            max_val = edge_mag[i].max()
            if max_val > 1e-8:
                edge_norm[i] = edge_mag[i] / max_val
        
        # Weight: 1 + λ * edge_norm
        return 1 + self.lambda_edge * edge_norm
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W)
            mask: Optional mask tensor (B, C, H, W)
        """
        # 1. Edge-weighted L1 loss
        edge_weight = self._compute_edge_weight(target)
        l1_per_pixel = torch.abs(pred - target)
        
        if mask is not None:
            mask_binary = (mask > 0.5).float()
            # Combine edge weight with mask weight
            weight_map = edge_weight * (mask_binary + (1 - mask_binary) * self.mask_outside_weight)
            loss_l1 = (l1_per_pixel * weight_map).sum() / (weight_map.sum() + 1e-8)
        else:
            loss_l1 = (l1_per_pixel * edge_weight).mean()
        
        # 2. SSIM loss (critical for SSIM metric!)
        loss_ssim = self.ssim_loss(pred, target)
        
        # 3. Gradient loss (edge preservation)
        loss_grad = self.edge_loss(pred, target, mask)
        
        # Combined
        total_loss = (
            self.l1_weight * loss_l1 +
            self.ssim_weight * loss_ssim +
            self.grad_weight * loss_grad
        )
        
        return total_loss


# ==============================================================================
# Training
# ==============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loss(config):
    """
    Create loss function based on config.loss_type.
    
    Options:
    - "combined": CombinedLoss (L1 + SSIM)
    - "masked": MaskedCombinedLoss (L1 + SSIM with mask weighting)
    - "edge_aware": EdgeAwareLoss (L1 + SSIM + Sobel, default)
    - "optimized": OptimizedLoss (L1 + SSIM + Grad + TV)
    - "edge_weighted": EdgeWeightedLoss (per-pixel edge weighting)
    """
    loss_type = getattr(config, 'loss_type', 'edge_aware')
    
    if loss_type == "combined":
        criterion = CombinedLoss(
            l1_weight=config.l1_weight,
            ssim_weight=config.ssim_weight
        )
        print(f"Using CombinedLoss (L1={config.l1_weight}, SSIM={config.ssim_weight})")
    
    elif loss_type == "masked":
        criterion = MaskedCombinedLoss(
            l1_weight=config.l1_weight,
            ssim_weight=config.ssim_weight,
            inside_weight=1.0,
            outside_weight=config.mask_outside_weight
        )
        print(f"Using MaskedCombinedLoss (outside_weight={config.mask_outside_weight})")
    
    elif loss_type == "optimized":
        criterion = OptimizedLoss(
            l1_weight=config.l1_weight,
            ssim_weight=config.ssim_weight,
            grad_weight=config.grad_weight,
            tv_weight=config.tv_weight,
            mask_outside_weight=config.mask_outside_weight
        )
        print(f"Using OptimizedLoss (L1={config.l1_weight}, SSIM={config.ssim_weight}, "
              f"Grad={config.grad_weight}, TV={config.tv_weight})")
    
    elif loss_type == "edge_weighted":
        criterion = EdgeWeightedLoss(
            l1_weight=config.l1_weight,
            ssim_weight=config.ssim_weight,
            grad_weight=config.grad_weight,
            lambda_edge=config.lambda_edge,
            mask_outside_weight=config.mask_outside_weight
        )
        print(f"Using EdgeWeightedLoss (L1={config.l1_weight}, SSIM={config.ssim_weight}, "
              f"Grad={config.grad_weight}, λ_edge={config.lambda_edge})")
    
    else:  # Default: edge_aware
        criterion = EdgeAwareLoss(
            l1_weight=config.l1_weight,
            ssim_weight=config.ssim_weight,
            edge_weight=config.edge_weight,
            mask_outside_weight=config.mask_outside_weight
        )
        print(f"Using EdgeAwareLoss (edge_weight={config.edge_weight})")
    
    return criterion


def calculate_ssim_masked(pred, target, mask, data_range=255):
    """
    Calculate SSIM only on masked region (LB-aligned).
    
    Args:
        pred: Predicted image (H, W), uint8
        target: Ground truth image (H, W), uint8
        mask: Binary mask (H, W), >0 = evaluate
        data_range: Max value (255 for uint8)
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mask_bool = mask > 0
    
    if mask_bool.sum() == 0:
        return 0.0
    
    pred_m = pred[mask_bool]
    target_m = target[mask_bool]
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    mu_p = pred_m.mean()
    mu_t = target_m.mean()
    
    sigma_p_sq = pred_m.var()
    sigma_t_sq = target_m.var()
    sigma_pt = ((pred_m - mu_p) * (target_m - mu_t)).mean()
    
    ssim_val = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p_sq + sigma_t_sq + C2))
    
    return float(ssim_val)


def calculate_psnr_masked(pred, target, mask, data_range=255):
    """
    Calculate PSNR only on masked region (LB-aligned).
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mask_bool = mask > 0
    
    if mask_bool.sum() == 0:
        return 0.0
    
    pred_m = pred[mask_bool]
    target_m = target[mask_bool]
    
    mse = np.mean((pred_m - target_m) ** 2)
    
    if mse == 0:
        return 100.0
    
    psnr_val = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr_val)


def calculate_metrics(pred, target, mask=None):
    """
    Calculate SSIM and PSNR for a batch (LB-aligned: uint8, data_range=255).
    
    Args:
        pred: Predicted tensor (B, 1, H, W), float [0, 1]
        target: Target tensor (B, 1, H, W), float [0, 1]
        mask: Optional mask tensor (B, 1, H, W), binary
    """
    # Convert to uint8 [0, 255] for LB-aligned evaluation
    pred_np = (pred.cpu().numpy() * 255).astype(np.uint8)
    target_np = (target.cpu().numpy() * 255).astype(np.uint8)
    
    if mask is not None:
        mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8)
    
    ssim_scores = []
    psnr_scores = []
    
    for i in range(pred_np.shape[0]):
        p = pred_np[i, 0]
        t = target_np[i, 0]
        
        if mask is not None and mask_np.shape[0] > i:
            m = mask_np[i, 0]
            ssim_scores.append(calculate_ssim_masked(p, t, m, data_range=255))
            psnr_scores.append(calculate_psnr_masked(p, t, m, data_range=255))
        else:
            # Fallback: full image evaluation with data_range=255
            ssim_scores.append(ssim(t, p, data_range=255))
            psnr_scores.append(psnr(t, p, data_range=255))
    
    return np.mean(ssim_scores), np.mean(psnr_scores)


def analyze_prediction_distribution(predictions, targets, masks, sample_ids):
    """
    Analyze prediction distribution for LB optimization.
    
    exp_013: Key diagnostic tool for systematic optimization.
    
    Returns dict with:
    - Global distribution stats
    - Per-sample stats  
    - Pred vs Target comparison
    - Diagnosis for next steps
    """
    stats = {
        'pred_inside': {'means': [], 'stds': []},
        'target_inside': {'means': [], 'stds': []},
        'per_sample': [],
    }
    
    all_pred_inside = []
    all_target_inside = []
    
    for pred, target, mask, sid in zip(predictions, targets, masks, sample_ids):
        mask_bool = mask > 0
        
        if mask_bool.sum() == 0:
            continue
        
        pred_inside = pred[mask_bool]
        target_inside = target[mask_bool]
        
        all_pred_inside.extend(pred_inside.tolist())
        all_target_inside.extend(target_inside.tolist())
        
        sample_stat = {
            'id': sid,
            'pred_mean': float(pred_inside.mean()),
            'pred_std': float(pred_inside.std()),
            'target_mean': float(target_inside.mean()),
            'target_std': float(target_inside.std()),
            'mean_diff': float(pred_inside.mean() - target_inside.mean()),
        }
        stats['per_sample'].append(sample_stat)
        stats['pred_inside']['means'].append(sample_stat['pred_mean'])
        stats['pred_inside']['stds'].append(sample_stat['pred_std'])
        stats['target_inside']['means'].append(sample_stat['target_mean'])
        stats['target_inside']['stds'].append(sample_stat['target_std'])
    
    # Global stats
    all_pred = np.array(all_pred_inside)
    all_target = np.array(all_target_inside)
    
    stats['global'] = {
        'pred_mean': float(all_pred.mean()) if len(all_pred) > 0 else 0,
        'pred_std': float(all_pred.std()) if len(all_pred) > 0 else 0,
        'target_mean': float(all_target.mean()) if len(all_target) > 0 else 0,
        'target_std': float(all_target.std()) if len(all_target) > 0 else 0,
        'mean_diff': float(all_pred.mean() - all_target.mean()) if len(all_pred) > 0 else 0,
        'std_ratio': float(all_pred.std() / all_target.std()) if len(all_target) > 0 and all_target.std() > 0 else 1,
        'sample_mean_variance': float(np.std(stats['pred_inside']['means'])) if len(stats['pred_inside']['means']) > 0 else 0,
    }
    
    return stats


def print_distribution_analysis(stats):
    """Print distribution analysis summary."""
    print(f"\n{'='*60}")
    print("Distribution Analysis (exp_013)")
    print(f"{'='*60}")
    
    g = stats['global']
    print(f"\nGlobal Stats (mask-inside pixels):")
    print(f"  Pred:   mean={g['pred_mean']:.1f}, std={g['pred_std']:.1f}")
    print(f"  Target: mean={g['target_mean']:.1f}, std={g['target_std']:.1f}")
    print(f"  Mean diff: {g['mean_diff']:.1f}")
    print(f"  Std ratio: {g['std_ratio']:.3f}")
    print(f"  Sample-to-sample variance: {g['sample_mean_variance']:.1f}")
    
    print(f"\n--- Diagnosis ---")
    
    # Mean偏移チェック
    if abs(g['mean_diff']) > 10:
        print(f"⚠️ Mean偏移が大きい ({g['mean_diff']:.1f}) → 分布正規化(mean matching)推奨")
    else:
        print(f"✅ Mean偏移は許容範囲 ({g['mean_diff']:.1f})")
    
    # Std比チェック
    if abs(g['std_ratio'] - 1) > 0.2:
        print(f"⚠️ Std比が不均衡 ({g['std_ratio']:.3f}) → コントラスト調整推奨")
    else:
        print(f"✅ Std比は許容範囲 ({g['std_ratio']:.3f})")
    
    # Sample間ばらつきチェック
    if g['sample_mean_variance'] > 20:
        print(f"⚠️ Sample間のばらつきが大きい ({g['sample_mean_variance']:.1f}) → 分布安定化が必要")
    else:
        print(f"✅ Sample間のばらつきは許容範囲 ({g['sample_mean_variance']:.1f})")
    
    print(f"{'='*60}")


def train_epoch(model, loader, criterion, optimizer, device):
    """Training epoch with optional mask-based loss weighting."""
    model.train()
    total_loss = 0
    
    # Check if criterion supports mask parameter
    # All custom losses except CombinedLoss support mask
    use_mask = isinstance(criterion, (MaskedCombinedLoss, EdgeAwareLoss, OptimizedLoss, 
                                       EdgeWeightedLoss, SobelEdgeLoss))
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        masks = batch.get("mask", None)
        if masks is not None:
            masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = torch.clamp(model(inputs), 0, 1)
        
        # Use MaskedCombinedLoss if available, otherwise standard loss
        if use_mask and masks is not None:
            loss = criterion(outputs, targets, masks)
        else:
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(loader)


def train_epoch_weighted(model, loader, criterion, optimizer, device):
    """Training epoch with sample-wise loss weighting for v5.
    
    Supports EdgeAwareLoss by passing masks to criterion.
    """
    model.train()
    total_loss = 0
    total_weighted_loss = 0
    
    # Check if criterion supports mask parameter
    use_mask = isinstance(criterion, (MaskedCombinedLoss, EdgeAwareLoss, OptimizedLoss, 
                                       EdgeWeightedLoss, SobelEdgeLoss))
    
    pbar = tqdm(loader, desc="Training (weighted)")
    for batch in pbar:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        weights = batch["weight"].to(device)  # Sample weights
        masks = batch.get("mask", None)
        if masks is not None:
            masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = torch.clamp(model(inputs), 0, 1)
        
        # Compute per-sample loss and apply weights
        batch_size = inputs.size(0)
        sample_losses = []
        for i in range(batch_size):
            if use_mask and masks is not None:
                sample_mask = masks[i:i+1]
                sample_loss = criterion(outputs[i:i+1], targets[i:i+1], sample_mask)
            else:
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
    
    # Check if criterion supports mask parameter
    use_mask = isinstance(criterion, (MaskedCombinedLoss, EdgeAwareLoss, OptimizedLoss, 
                                       EdgeWeightedLoss, SobelEdgeLoss))
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            masks = batch.get("mask", None)
            if masks is not None:
                masks = masks.to(device)
            
            outputs = torch.clamp(model(inputs), 0, 1)
            
            # Pass mask to criterion if supported
            if use_mask and masks is not None:
                loss = criterion(outputs, targets, masks)
            else:
                loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # LB-aligned metrics with mask
            batch_ssim, batch_psnr = calculate_metrics(outputs, targets, masks)
            total_ssim += batch_ssim
            total_psnr += batch_psnr
            n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "ssim": total_ssim / n_batches,
        "psnr": total_psnr / n_batches,
    }


def validate_with_categories(model, loader, criterion, device, config=None):
    """Validation with category-wise metrics and distribution analysis."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    category_metrics = {
        'A': {'ssim': [], 'psnr': []},
        'B': {'ssim': [], 'psnr': []},
        'C': {'ssim': [], 'psnr': []},
    }
    
    # exp_013: Distribution analysis data
    analyze_dist = config is not None and getattr(config, 'analyze_distribution', False)
    dist_data = {'predictions': [], 'targets': [], 'masks': [], 'sample_ids': []}
    
    # Check if criterion supports mask parameter
    use_mask = isinstance(criterion, (MaskedCombinedLoss, EdgeAwareLoss, OptimizedLoss, 
                                       EdgeWeightedLoss, SobelEdgeLoss))
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            categories = batch["category"]
            sample_ids = batch.get("id", [f"sample_{i}" for i in range(len(categories))])
            masks = batch.get("mask", None)
            if masks is not None:
                masks = masks.to(device)
            
            outputs = torch.clamp(model(inputs), 0, 1)
            
            # Pass mask to criterion if supported
            if use_mask and masks is not None:
                loss = criterion(outputs, targets, masks)
            else:
                loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1
            
            # Convert to uint8 for LB-aligned evaluation
            pred_np = (outputs.cpu().numpy() * 255).astype(np.uint8)
            target_np = (targets.cpu().numpy() * 255).astype(np.uint8)
            mask_np = None
            if masks is not None:
                mask_np = (masks.cpu().numpy() > 0.5).astype(np.uint8)
            
            for i, cat in enumerate(categories):
                p = pred_np[i, 0]
                t = target_np[i, 0]
                m = mask_np[i, 0] if mask_np is not None and mask_np.shape[0] > i else None
                
                if m is not None:
                    s = calculate_ssim_masked(p, t, m, data_range=255)
                    pn = calculate_psnr_masked(p, t, m, data_range=255)
                else:
                    s = ssim(t, p, data_range=255)
                    pn = psnr(t, p, data_range=255)
                
                if cat in category_metrics:
                    category_metrics[cat]['ssim'].append(s)
                    category_metrics[cat]['psnr'].append(pn)
                
                # exp_013: Collect distribution data
                if analyze_dist and m is not None:
                    dist_data['predictions'].append(p)
                    dist_data['targets'].append(t)
                    dist_data['masks'].append(m)
                    dist_data['sample_ids'].append(sample_ids[i] if i < len(sample_ids) else f"sample_{i}")
    
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
    
    # exp_013: Run distribution analysis
    if analyze_dist and len(dist_data['predictions']) > 0:
        dist_stats = analyze_prediction_distribution(
            dist_data['predictions'],
            dist_data['targets'],
            dist_data['masks'],
            dist_data['sample_ids']
        )
        print_distribution_analysis(dist_stats)
        results['distribution'] = dist_stats['global']
    
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
    
    # Loss and optimizer (use create_loss for configurable loss selection)
    criterion = create_loss(config).to(config.device)  # Move to GPU for Sobel buffers
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
    
    # Load full dataframe and exclude problematic samples
    df = pd.read_csv(config.train_csv)
    df = filter_excluded_samples(df)
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
        
        # Initialize Trackio for this fold
        if TRACKIO_AVAILABLE and getattr(config, 'tracking_enabled', False):
            experiment_name = os.environ.get("EXPERIMENT_ID", "exp_unknown")
            trackio.init(
                project=config.tracking_project,
                name=f"{experiment_name}_fold{fold}",
                config={
                    "architecture": getattr(config, 'architecture', 'unet'),
                    "encoder": config.encoder,
                    "loss_type": getattr(config, 'loss_type', 'combined'),
                    "learning_rate": config.learning_rate,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "fold": fold,
                    "n_folds": n_folds,
                    "image_size": config.image_size,
                    "augmentation_enabled": getattr(config, 'augmentation_enabled', False),
                }
            )
        
        # exp_023: Create augmentation for training (geometric-only mode)
        train_aug = None
        if getattr(config, 'augmentation_enabled', False):
            aug_strength = getattr(config, 'augmentation_strength', 0.5)
            aug_mode = getattr(config, 'augmentation_mode', 'geometric')
            train_aug = get_training_augmentation(strength=aug_strength, mode=aug_mode)
            if fold == 0:  # Print only once
                print(f"Augmentation enabled (strength={aug_strength}, mode={aug_mode})")
        
        # Create datasets with indices
        train_dataset = OrganoidDataset(
            df, config.data_dir, config.image_size, is_test=False, 
            indices=train_idx, augmentation=train_aug
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
        
        criterion = create_loss(config).to(config.device)  # Move to GPU for Sobel buffers
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
            val_metrics = validate_with_categories(model, val_loader, criterion, config.device, config)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val SSIM: {val_metrics['ssim']:.4f} (A:{val_metrics['ssim_A']:.4f}, B:{val_metrics['ssim_B']:.4f}, C:{val_metrics['ssim_C']:.4f})")
            print(f"Val PSNR: {val_metrics['psnr']:.2f} (A:{val_metrics['psnr_A']:.2f}, B:{val_metrics['psnr_B']:.2f}, C:{val_metrics['psnr_C']:.2f})")
            
            # Log metrics to Trackio
            if TRACKIO_AVAILABLE and getattr(config, 'tracking_enabled', False):
                lb_score = calculate_lb_score(val_metrics['ssim'], val_metrics['psnr'])
                trackio.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/ssim": val_metrics['ssim'],
                    "val/psnr": val_metrics['psnr'],
                    "val/lb_score": lb_score,
                    "val/ssim_A": val_metrics['ssim_A'],
                    "val/ssim_B": val_metrics['ssim_B'],
                    "val/ssim_C": val_metrics['ssim_C'],
                    "val/ssim_C_worst20": val_metrics.get('ssim_C_worst20', 0),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                })
            
            if val_metrics['ssim'] > best_fold_ssim:
                best_fold_ssim = val_metrics['ssim']
                # Save best model for this fold
                torch.save(model.state_dict(), config.output_dir / f"best_model_fold{fold}.pth")
        
        # Store fold results
        final_metrics = validate_with_categories(model, val_loader, criterion, config.device, config)
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
        
        # Log fold summary to Trackio
        if TRACKIO_AVAILABLE and getattr(config, 'tracking_enabled', False):
            trackio.log({
                f"fold{fold}/best_ssim": best_fold_ssim,
                f"fold{fold}/final_ssim": final_metrics['ssim'],
                f"fold{fold}/final_psnr": final_metrics['psnr'],
            })
            trackio.finish()
        
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
    
    # Load full dataframe and exclude problematic samples
    df = pd.read_csv(config.train_csv)
    df = filter_excluded_samples(df)
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
        
        criterion = create_loss(config).to(config.device)  # Move to GPU for Sobel buffers
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
            val_metrics = validate_with_categories(model, val_loader, criterion, config.device, config)
            
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
        final_val_metrics = validate_with_categories(model, val_loader, criterion, config.device, config)
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
    
    # Load full dataframe and exclude problematic samples
    df = pd.read_csv(config.train_csv)
    df = filter_excluded_samples(df)
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
        
        # exp_022: Create augmentation for training
        train_aug = None
        if getattr(config, 'augmentation_enabled', False):
            aug_strength = getattr(config, 'augmentation_strength', 0.5)
            train_aug = get_training_augmentation(strength=aug_strength)
            if fold == 0:  # Print only once
                print(f"Augmentation enabled (strength={aug_strength})")
        
        # Create datasets with sample weights
        train_df = df.loc[train_all_idx]
        val_df = df.loc[val_original_idx]
        
        train_dataset = OrganoidDataset(
            train_df, config.data_dir, config.image_size, 
            is_test=False, sample_weights=sample_weights, augmentation=train_aug
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
        criterion = create_loss(config).to(config.device)  # Move to GPU for Sobel buffers
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
            val_metrics = validate_with_categories(model, val_loader, criterion, config.device, config)
            
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
        final_val = validate_with_categories(model, val_loader, criterion, config.device, config)
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
# Pseudo-Labeling Pipeline (exp_030)
# ==============================================================================

def generate_pseudo_labels(config):
    """
    Generate pseudo-labels for test data using Phase 1 fold ensemble.
    
    Process:
    1. Load all fold models from Phase 1
    2. For each test image: TTA + weighted fold ensemble prediction
    3. Return predictions as numpy arrays (H, W) float32 [0, 1]
    """
    print(f"\n{'='*60}")
    print("Generating Pseudo-Labels for Test Data")
    print(f"{'='*60}")
    
    import cv2
    
    # Load fold models
    cv_results_path = config.output_dir / "cv_results.json"
    if not cv_results_path.exists():
        print("ERROR: cv_results.json not found. Cannot generate pseudo-labels.")
        return None
    
    with open(cv_results_path, 'r') as f:
        cv_results = json.load(f)
    
    fold_results = cv_results.get('fold_results', [])
    fold_data = []
    for fold_result in fold_results:
        fold_idx = fold_result['fold'] - 1
        fold_ssim = fold_result['ssim']
        fold_model_path = config.output_dir / f"best_model_fold{fold_idx}.pth"
        if fold_model_path.exists():
            fold_data.append({'idx': fold_idx, 'ssim': fold_ssim, 'path': fold_model_path})
    
    if not fold_data:
        print("ERROR: No fold models found.")
        return None
    
    # Sort by SSIM and use rank-based weights
    fold_data.sort(key=lambda x: x['ssim'], reverse=True)
    rank_weights = getattr(config, 'fold_rank_weights', [1.0, 0.9, 0.8, 0.7, 0.6])
    weights = np.array(rank_weights[:len(fold_data)])
    weights = weights / weights.sum()
    
    print(f"Using {len(fold_data)} fold models for pseudo-label generation:")
    for i, fd in enumerate(fold_data):
        print(f"  Fold {fd['idx']}: SSIM={fd['ssim']:.4f}, weight={weights[i]:.3f}")
    
    # Load models
    models = []
    for fd in fold_data:
        model = create_model(config)
        model.load_state_dict(torch.load(fd['path'], map_location=config.device))
        model.eval()
        models.append(model)
    
    # Load test data
    test_df = pd.read_csv(config.test_csv)
    test_dataset = OrganoidDataset(test_df, config.data_dir, config.image_size, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    pseudo_labels = {}  # id -> numpy array (H, W) float32 [0, 1]
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating pseudo-labels"):
            inputs = batch["input"].to(config.device)
            ids = batch["id"]
            
            # Ensemble prediction with TTA
            ensemble_preds = []
            for model in models:
                tta_mode = getattr(config, 'tta_mode', "flip4")
                tta_aggregate = getattr(config, 'tta_aggregate', "median")
                pred = predict_with_tta(model, inputs, config.device, mode=tta_mode, aggregate=tta_aggregate)
                pred = torch.clamp(pred, 0, 1)
                ensemble_preds.append(pred)
            
            # Weighted average across folds
            stacked = torch.stack(ensemble_preds, dim=0)
            weights_tensor = torch.tensor(weights, device=config.device, dtype=torch.float32)
            weights_tensor = weights_tensor.view(-1, 1, 1, 1, 1)
            final_pred = (stacked * weights_tensor).sum(dim=0)
            
            for i, sample_id in enumerate(ids):
                pseudo_labels[sample_id] = final_pred[i, 0].cpu().numpy()
    
    print(f"Generated pseudo-labels for {len(pseudo_labels)} test samples")
    
    # Cleanup models to free GPU memory
    del models
    torch.cuda.empty_cache()
    
    return pseudo_labels


class PseudoLabelDataset(Dataset):
    """Dataset that combines real training data with pseudo-labeled test data."""
    
    def __init__(self, real_df, pseudo_df, data_dir, image_size, pseudo_labels,
                 pseudo_weight=0.5, augmentation=None):
        """
        Args:
            real_df: DataFrame of real training samples
            pseudo_df: DataFrame of test samples (for input images)
            data_dir: Path to data directory
            image_size: Target image size
            pseudo_labels: dict of {sample_id: numpy array (H, W) float32 [0, 1]}
            pseudo_weight: Loss weight for pseudo-labeled samples
            augmentation: Albumentations augmentation pipeline
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augmentation = augmentation
        
        # Build combined sample list
        self.samples = []
        
        # Real samples (weight = 1.0) — store CSV row for correct path resolution
        for _, row in real_df.iterrows():
            self.samples.append({
                'id': row['id'],
                'input_path': row['input_path'],       # e.g. "train/train_00000.png"
                'target_path': row.get('target_path'),  # e.g. "train/train_00000_target.png"
                'mask_path': row.get('mask_path'),      # e.g. "train/train_00000_mask.png"
                'is_pseudo': False,
                'weight': 1.0,
            })
        
        # Pseudo samples (weight = pseudo_weight) — use CSV input_path from test.csv
        for _, row in pseudo_df.iterrows():
            sample_id = row['id']
            if sample_id in pseudo_labels:
                self.samples.append({
                    'id': sample_id,
                    'input_path': row['input_path'],    # e.g. "test/test_00000.png"
                    'target_path': None,                # pseudo-labeled (no ground truth)
                    'mask_path': None,
                    'is_pseudo': True,
                    'weight': pseudo_weight,
                    'pseudo_target': pseudo_labels[sample_id],
                })
        
        # CLAHE config
        self.clahe_enabled = getattr(Config, 'clahe_enabled', True)
        self.clahe_clip_limit = getattr(Config, 'clahe_clip_limit', 2.0)
        self.clahe_tile_size = getattr(Config, 'clahe_tile_size', (8, 8))
        
        print(f"PseudoLabelDataset: {sum(1 for s in self.samples if not s['is_pseudo'])} real + "
              f"{sum(1 for s in self.samples if s['is_pseudo'])} pseudo = {len(self.samples)} total")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['id']
        
        # Load input image using CSV path (matches OrganoidDataset behavior)
        input_path = self.data_dir / sample['input_path']
        
        img = np.array(Image.open(input_path).convert('L'))
        
        # CLAHE preprocessing
        if self.clahe_enabled:
            import cv2
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size
            )
            img = clahe.apply(img)
        
        # Resize if needed
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            import cv2
            img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Get target
        if sample['is_pseudo']:
            target = (sample['pseudo_target'] * 255).astype(np.uint8)
        else:
            # Use CSV target_path (e.g. "train/train_00000_target.png")
            target_path = self.data_dir / sample['target_path']
            target = np.array(Image.open(target_path).convert('L'))
            if target.shape[0] != self.image_size or target.shape[1] != self.image_size:
                import cv2
                target = cv2.resize(target, (self.image_size, self.image_size))
        
        # Load mask using CSV mask_path
        mask = None
        if sample['mask_path'] and pd.notna(sample['mask_path']):
            mask_path = self.data_dir / sample['mask_path']
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert('L'))
                if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
                    import cv2
                    mask = cv2.resize(mask, (self.image_size, self.image_size))
        if mask is None:
            mask = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 255
        
        # Apply augmentation
        if self.augmentation is not None:
            augmented = self.augmentation(image=img, target=target, mask=mask)
            img = augmented['image']
            target = augmented['target']
            mask = augmented['mask']
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        target_tensor = torch.from_numpy(target.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        
        return {
            "input": img_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "id": sample_id,
            "weight": sample['weight'],
        }


def pseudo_label_finetune(config, pseudo_labels):
    """
    Phase 2: Fine-tune each fold model on real + pseudo-labeled data.
    
    Key design decisions:
    - Lower LR (1/3 of Phase 1) to avoid catastrophic forgetting
    - Pseudo samples weighted at 0.5x (lower confidence)
    - Shorter training (10 epochs by default)
    """
    print(f"\n{'='*60}")
    print("Phase 2: Pseudo-Label Fine-tuning")
    print(f"{'='*60}")
    
    phase2_epochs = getattr(config, 'pseudo_label_epochs', 10)
    phase2_lr = config.learning_rate * getattr(config, 'pseudo_label_lr_factor', 0.3)
    pseudo_weight = getattr(config, 'pseudo_label_weight', 0.5)
    
    print(f"Phase 2 LR: {phase2_lr}")
    print(f"Phase 2 Epochs: {phase2_epochs}")
    print(f"Pseudo sample weight: {pseudo_weight}")
    
    # Load real training data
    df = pd.read_csv(config.train_csv)
    df = filter_excluded_samples(df)
    
    # Load test data info
    test_df = pd.read_csv(config.test_csv)
    
    # Create augmentation
    train_aug = None
    if getattr(config, 'augmentation_enabled', False):
        aug_strength = getattr(config, 'augmentation_strength', 0.6)
        aug_mode = getattr(config, 'augmentation_mode', 'geometric')
        train_aug = get_training_augmentation(strength=aug_strength, mode=aug_mode)
    
    # Stratified K-Fold setup (same as Phase 1 for consistency)
    df = cluster_category_c(df, config.data_dir, n_clusters=3)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['difficulty'])):
        fold_model_path = config.output_dir / f"best_model_fold{fold}.pth"
        if not fold_model_path.exists():
            print(f"Fold {fold}: Model not found, skipping")
            continue
        
        print(f"\n{'='*40}")
        print(f"Phase 2 - Fold {fold + 1}/5")
        print(f"{'='*40}")
        
        # Load Phase 1 model
        model = create_model(config)
        model.load_state_dict(torch.load(fold_model_path, map_location=config.device))
        
        # Create combined dataset (real train + pseudo test)
        train_df = df.loc[train_idx]
        combined_dataset = PseudoLabelDataset(
            real_df=train_df,
            pseudo_df=test_df,
            data_dir=config.data_dir,
            image_size=config.image_size,
            pseudo_labels=pseudo_labels,
            pseudo_weight=pseudo_weight,
            augmentation=train_aug,
        )
        
        # Validation set stays the same (real data only)
        val_dataset = OrganoidDataset(
            df, config.data_dir, config.image_size, is_test=False, indices=val_idx
        )
        
        train_loader = DataLoader(
            combined_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers, pin_memory=True
        )
        
        criterion = create_loss(config).to(config.device)
        optimizer = optim.AdamW(model.parameters(), lr=phase2_lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, phase2_epochs)
        
        best_ssim = 0
        
        for epoch in range(phase2_epochs):
            # Training with sample weighting
            model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                inputs = batch["input"].to(config.device)
                targets = batch["target"].to(config.device)
                masks = batch["mask"].to(config.device)
                weights = batch["weight"].to(config.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Compute per-sample loss with mask
                if hasattr(criterion, 'forward') and 'mask' in criterion.forward.__code__.co_varnames:
                    loss = criterion(outputs, targets, masks)
                else:
                    loss = criterion(outputs, targets)
                
                # Weight by sample confidence (real=1.0, pseudo=0.5)
                # This is a simplified approach - weight the total batch loss
                batch_weight = weights.mean()
                weighted_loss = loss * batch_weight
                
                weighted_loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            
            # Validation
            val_results = validate(model, val_loader, criterion, config.device)
            val_ssim = val_results['ssim']
            
            print(f"  Epoch {epoch+1}/{phase2_epochs} - "
                  f"Train Loss: {epoch_loss/n_batches:.4f}, "
                  f"Val SSIM: {val_ssim:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model (overwrite Phase 1)
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save(model.state_dict(), fold_model_path)
                print(f"  → Saved best Phase 2 model (SSIM: {best_ssim:.4f})")
        
        print(f"Fold {fold+1} Phase 2 complete. Best SSIM: {best_ssim:.4f}")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("Phase 2 Pseudo-Label Fine-tuning Complete!")
    print(f"{'='*60}")


# ==============================================================================
# Inference and Submission
# ==============================================================================

def _build_tta_transforms(mode):
    """Return a list of (forward, inverse) TTA transforms."""
    if mode == "flip4":
        return [
            (lambda x: x, lambda x: x),
            (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
            (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
            (lambda x: torch.flip(x, dims=[2, 3]), lambda x: torch.flip(x, dims=[2, 3])),
        ]
    if mode == "dihedral8":
        transforms = []
        for k in range(4):
            rot = lambda x, k=k: torch.rot90(x, k, dims=[2, 3])
            inv_rot = lambda x, k=k: torch.rot90(x, (-k) % 4, dims=[2, 3])
            transforms.append((rot, inv_rot))
            rot_flip = lambda x, k=k: torch.flip(torch.rot90(x, k, dims=[2, 3]), dims=[3])
            inv_rot_flip = lambda x, k=k: torch.rot90(torch.flip(x, dims=[3]), (-k) % 4, dims=[2, 3])
            transforms.append((rot_flip, inv_rot_flip))
        return transforms
    print(f"Warning: Unknown tta_mode '{mode}', falling back to flip4")
    return _build_tta_transforms("flip4")


def predict_with_tta(model, inputs, device, mode="flip4", aggregate="median"):
    """
    Predict with Test Time Augmentation (TTA).
    
    Modes:
    - flip4: original + H/V/both flips
    - dihedral8: 90-degree rotations and horizontal flips
    
    Aggregate:
    - median (robust) or mean
    """
    predictions = []
    transforms = _build_tta_transforms(mode)
    
    for forward_t, inverse_t in transforms:
        with torch.no_grad():
            pred = model(forward_t(inputs))
            predictions.append(inverse_t(pred))
    
    stacked = torch.stack(predictions, dim=0)
    if aggregate == "mean":
        return stacked.mean(dim=0)
    return stacked.median(dim=0)[0]

def predict_and_submit(config, model_path=None):
    """
    Run inference on test set and create submission CSV.
    
    exp_023: Weighted Fold Ensemble
    - Load all fold models (best_model_fold{i}.pth)
    - Each model: TTA with median (robust to geometric outliers)
    - Across folds: weighted mean by validation SSIM
    
    Submission format:
    - 512×512 image flattened to 262,144 pixels
    - CSV columns: id, pixel_0, pixel_1, ..., pixel_262143
    - Values: 0-255 (uint8)
    """
    print(f"\n{'='*60}")
    print("Running Inference on Test Set (Weighted Fold Ensemble)")
    print(f"{'='*60}")
    
    import cv2  # For resizing
    
    # Load fold weights from cv_results.json
    cv_results_path = config.output_dir / "cv_results.json"
    fold_models = []
    fold_weights = []
    
    if cv_results_path.exists():
        with open(cv_results_path, 'r') as f:
            cv_results = json.load(f)
        
        # Extract SSIM scores for fold selection
        fold_results = cv_results.get('fold_results', [])
        fold_data = []
        
        for fold_result in fold_results:
            fold_idx = fold_result['fold'] - 1  # 0-indexed
            fold_ssim = fold_result['ssim']
            fold_model_path = config.output_dir / f"best_model_fold{fold_idx}.pth"
            
            if fold_model_path.exists():
                fold_data.append({
                    'idx': fold_idx,
                    'ssim': fold_ssim,
                    'path': fold_model_path
                })
                print(f"  Fold {fold_idx}: SSIM={fold_ssim:.4f} -> {fold_model_path.name}")
            else:
                print(f"  Fold {fold_idx}: Model not found at {fold_model_path}")
        
        # exp_025: Top-3 Fold Selection (critical change)
        # Sort by SSIM descending and take top 3
        fold_data.sort(key=lambda x: x['ssim'], reverse=True)
        
        n_select = getattr(config, 'n_folds_ensemble', 3)  # Default: top 3
        selected_folds = fold_data[:n_select]
        
        print(f"\n  exp_025: Selecting top {n_select} folds (noise reduction)")
        for i, fold in enumerate(selected_folds):
            print(f"    Rank {i+1}: Fold {fold['idx']} (SSIM={fold['ssim']:.4f})")
        
        # Rejected folds
        rejected_folds = fold_data[n_select:]
        if rejected_folds:
            print(f"  Rejected {len(rejected_folds)} weak folds:")
            for fold in rejected_folds:
                print(f"    Fold {fold['idx']} (SSIM={fold['ssim']:.4f}) - EXCLUDED")
        
        fold_models = [f['path'] for f in selected_folds]
        
        # exp_025: Rank-based weights (NOT softmax)
        # Best = 1.0, Mid = 0.7, Low = 0.4 (or custom from config)
        rank_weights = getattr(config, 'fold_rank_weights', [1.0, 0.7, 0.4])
        fold_weights = rank_weights[:len(fold_models)]
    else:
        print(f"  cv_results.json not found at {cv_results_path}")
    
    # Fallback to single best model if no fold models found
    if not fold_models:
        fallback_path = model_path if model_path else config.output_dir / "best_model.pth"
        if fallback_path.exists():
            fold_models = [fallback_path]
            fold_weights = [1.0]
            print(f"  Fallback: Using single model: {fallback_path}")
        else:
            print(f"  ERROR: No models found! Fallback path: {fallback_path}")
            return None
    
    # Normalize rank-based weights to sum to 1
    fold_weights = np.array(fold_weights)
    fold_weights = fold_weights / fold_weights.sum()
    
    print(f"\nexp_025 Fold Weights (rank-based, normalized):")
    for i, (mpath, weight) in enumerate(zip(fold_models, fold_weights)):
        print(f"  Rank {i+1}: weight={weight:.4f}")
    
    # Load all models
    models = []
    for model_path in fold_models:
        model = create_model(config)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        models.append(model)
    print(f"\nLoaded {len(models)} models for ensemble")
    
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
    
    # Collect predictions
    all_ids = []
    all_pixels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference (Fold Ensemble)"):
            inputs = batch["input"].to(config.device)
            ids = batch["id"]
            batch_size = inputs.shape[0]
            
            # Collect predictions from all models
            ensemble_preds = []
            for model_idx, model in enumerate(models):
                # Each model uses TTA with median (geometric outlier robust)
                if getattr(config, 'tta_enabled', True):  # Default to True for ensemble
                    tta_mode = getattr(config, 'tta_mode', "flip4")
                    tta_aggregate = getattr(config, 'tta_aggregate', "median")
                    model_pred = predict_with_tta(
                        model,
                        inputs,
                        config.device,
                        mode=tta_mode,
                        aggregate=tta_aggregate,
                    )
                else:
                    model_pred = model(inputs)
                model_pred = torch.clamp(model_pred, 0, 1)
                ensemble_preds.append(model_pred)
            
            # Weighted mean across folds
            # Stack: [n_models, batch, 1, H, W]
            stacked_preds = torch.stack(ensemble_preds, dim=0)
            
            # Apply weights: [n_models] -> [n_models, 1, 1, 1, 1]
            weights_tensor = torch.tensor(fold_weights, device=config.device, dtype=torch.float32)
            weights_tensor = weights_tensor.view(-1, 1, 1, 1, 1)
            
            # Weighted sum
            outputs = (stacked_preds * weights_tensor).sum(dim=0)
            
            for i, sample_id in enumerate(ids):
                # Get prediction as numpy array
                pred = outputs[i, 0].cpu().numpy()
                
                # Convert to uint8 (0-255)
                pred_uint8 = (pred * 255).astype(np.uint8)
                
                # Ensure exactly 512x512 (as per baseline)
                if pred_uint8.shape != (512, 512):
                    print(f"Resizing from {pred_uint8.shape} to (512, 512)")
                    pred_uint8 = cv2.resize(pred_uint8, (512, 512))
                
                # exp_014: Apply global mean matching
                if getattr(config, 'mean_matching_enabled', False):
                    delta = getattr(config, 'mean_matching_delta', 0.0)
                    pred_float = pred_uint8.astype(np.float32) + delta
                    pred_uint8 = np.clip(pred_float, 0, 255).astype(np.uint8)
                
                # Post-processing: Median filter (salt-pepper noise removal)
                median_size = getattr(config, 'median_filter_size', 0)
                if median_size > 0:
                    pred_uint8 = cv2.medianBlur(pred_uint8, median_size)
                
                # Post-processing: Unsharp mask (edge enhancement)
                unsharp_strength = getattr(config, 'unsharp_strength', 0.0)
                if unsharp_strength > 0:
                    unsharp_radius = getattr(config, 'unsharp_radius', 1)
                    blur_size = 2 * unsharp_radius + 1
                    blurred = cv2.GaussianBlur(pred_uint8, (blur_size, blur_size), 0)
                    pred_float = pred_uint8.astype(np.float32)
                    sharpened = pred_float + unsharp_strength * (pred_float - blurred.astype(np.float32))
                    pred_uint8 = np.clip(sharpened, 0, 255).astype(np.uint8)
                
                # Flatten to 1D (262,144 pixels) - Row-major ('C') order
                pixels_flat = pred_uint8.flatten()
                
                all_ids.append(sample_id)
                all_pixels.append(pixels_flat)
    
    # Create submission DataFrame
    print(f"\nCreating submission CSV...")
    print(f"Number of samples: {len(all_ids)}")
    print(f"Pixels per sample: {len(all_pixels[0]) if all_pixels else 0}")
    
    # Debug: show first sample info
    if all_pixels:
        first_pixels = all_pixels[0]
        print(f"First sample ID: {all_ids[0]}")
        print(f"First sample pixel stats: min={first_pixels.min()}, max={first_pixels.max()}, mean={first_pixels.mean():.1f}")
    
    # Column names: id, pixel_0, pixel_1, ..., pixel_262143
    n_pixels = 512 * 512  # 262,144
    pixel_columns = [f"pixel_{i}" for i in range(n_pixels)]
    
    submission_df = pd.DataFrame(all_pixels, columns=pixel_columns)
    submission_df.insert(0, "id", all_ids)
    
    # Save CSV
    csv_path = config.output_dir / "submission.csv"
    submission_df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"📄 Submission CSV created: {csv_path}")
    print(f"   Shape: {submission_df.shape}")
    print(f"   Size: {csv_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Ensemble: {len(models)} models with TTA median + weighted mean")
    print(f"   First row preview:")
    print(f"   id: {submission_df.iloc[0]['id']}")
    print(f"   pixel_0: {submission_df.iloc[0]['pixel_0']}")
    print(f"   pixel_262143: {submission_df.iloc[0]['pixel_262143']}")
    print(f"{'='*60}")
    
    return csv_path


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
    
    # Training (Phase 1)
    if cv_mode == "worst_case_v5":
        train_worst_case_cv_v5(config, n_folds=n_folds)
    elif cv_mode == "worst_case":
        train_worst_case_cv(config, n_folds=n_folds)
    elif cv_mode == "kfold" and n_folds > 1:
        train_kfold(config, n_folds=n_folds)
    else:
        train(config)
    
    # exp_030: Pseudo-Labeling (Phase 2)
    if getattr(config, 'pseudo_label_enabled', False):
        pseudo_labels = generate_pseudo_labels(config)
        if pseudo_labels is not None:
            pseudo_label_finetune(config, pseudo_labels)
            del pseudo_labels  # Free memory
            torch.cuda.empty_cache()
        else:
            print("WARNING: Pseudo-label generation failed. Skipping Phase 2.")
    
    # Inference (for LB submission)
    if run_inference:
        predict_and_submit(config)


