"""
Competition Training with VirtualStaining Transfer Learning
本番コンペ学習スクリプト（VirtualStaining encoder転移）
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Transfer learning module
from transfer_learning import (
    load_virtualstaining_encoder,
    create_transfer_model,
    create_differential_optimizer,
    reset_batchnorm_statistics
)


# ==============================================================================
# Config
# ==============================================================================

class Config:
    # Data
    data_dir = Path("medical-ai-contest-7th-2025")
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    
    # Transfer Learning
    pretrained_path = Path("notebooks/best_model.pth")
    use_transfer = True
    freeze_encoder = False  # 凍結はNG！
    
    # Learning rates (差分学習率)
    encoder_lr = 1e-5      # Encoder: 低速
    bottleneck_lr = 5e-5    # Bottleneck: 中速
    decoder_lr = 1e-4       # Decoder: 通常
    
    # Training
    image_size = 512
    epochs = 50
    batch_size = 4
    weight_decay = 1e-5
    
    # Loss
    mse_weight = 1.0
    ssim_weight = 1.0
    
    # CLAHE (VirtualStainingと同じ設定を維持)
    use_clahe = True
    clahe_clip_limit = 2.0
    clahe_tile_size = (8, 8)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("outputs/transfer_trained")
    seed = 42


# ==============================================================================
# CLAHE (VirtualStainingと同じ前処理を維持)
# ==============================================================================

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """VirtualStainingと同じCLAHE処理."""
    if len(image.shape) == 3:
        image = image[:, :, 0]
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


# ==============================================================================
# Dataset (本番コンペ用)
# ==============================================================================

class CompetitionDataset(Dataset):
    """本番コンペ用データセット."""
    
    def __init__(self, csv_path, data_dir, image_size=512, use_clahe=True, is_test=False):
        self.df = pd.read_csv(csv_path)
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.is_test = is_test
        
        # 除外サンプル
        excluded = {"train_00099", "train_00603", "train_00802", "train_00863"}
        if not is_test and "id" in self.df.columns:
            self.df = self.df[~self.df["id"].isin(excluded)].reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 入力画像
        input_path = self.data_dir / row["input_path"]
        input_img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        
        # CLAHE適用 (VirtualStaining学習時と同じ前処理)
        if self.use_clahe:
            input_img = apply_clahe(input_img)
        
        input_img = cv2.resize(input_img, (self.image_size, self.image_size))
        input_arr = input_img.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(input_arr).unsqueeze(0)
        
        if self.is_test:
            return {"id": row["id"], "input": input_tensor}
        
        # ターゲット
        target_path = self.data_dir / row["target_path"]
        target_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        target_img = cv2.resize(target_img, (self.image_size, self.image_size))
        target_arr = target_img.astype(np.float32) / 255.0
        target_tensor = torch.from_numpy(target_arr).unsqueeze(0)
        
        # マスク (評価用)
        mask_path = self.data_dir / row["mask_path"]
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask_arr = mask_img.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        
        return {
            "id": row["id"],
            "input": input_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
        }


# ==============================================================================
# Loss (VirtualStainingと同じ)
# ==============================================================================

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2*1.5**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        _1D = gauss.unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', _2D)
    
    def forward(self, img1, img2):
        window = self.window.to(img1.device)
        pad = self.window_size // 2
        
        mu1 = nn.functional.conv2d(img1, window, padding=pad)
        mu2 = nn.functional.conv2d(img2, window, padding=pad)
        
        sigma1_sq = nn.functional.conv2d(img1*img1, window, padding=pad) - mu1**2
        sigma2_sq = nn.functional.conv2d(img2*img2, window, padding=pad) - mu2**2
        sigma12 = nn.functional.conv2d(img1*img2, window, padding=pad) - mu1*mu2
        
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, ssim_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + self.ssim_weight * self.ssim(pred, target)


# ==============================================================================
# Training
# ==============================================================================

def train_with_transfer(config):
    """VirtualStaining転移学習を使った本番学習."""
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== Model Setup =====
    print("\n" + "="*60)
    print("Setting up Transfer Learning from VirtualStaining")
    print("="*60)
    
    # VirtualStaining encoder重みをロード
    encoder_weights = None
    if config.use_transfer and config.pretrained_path.exists():
        encoder_weights = load_virtualstaining_encoder(config.pretrained_path, config.device)
    else:
        print("⚠️ No pretrained weights found, training from scratch")
    
    # 転移学習モデルを作成
    model = create_transfer_model(
        encoder_weights=encoder_weights,
        freeze_encoder=config.freeze_encoder
    )
    model = model.to(config.device)
    
    # BatchNorm統計をリセット (重要!)
    reset_batchnorm_statistics(model)
    
    # 差分学習率オプティマイザ
    optimizer = create_differential_optimizer(
        model,
        encoder_lr=config.encoder_lr,
        bottleneck_lr=config.bottleneck_lr,
        decoder_lr=config.decoder_lr,
        weight_decay=config.weight_decay
    )
    
    # ===== Data =====
    print("\n" + "="*60)
    print("Loading Competition Dataset")
    print("="*60)
    
    full_dataset = CompetitionDataset(
        config.train_csv, config.data_dir, config.image_size, config.use_clahe
    )
    
    n_train = int(len(full_dataset) * 0.8)
    n_val = len(full_dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    # ===== Training =====
    criterion = CombinedLoss(config.mse_weight, config.ssim_weight)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    
    best_ssim = 0
    history = []
    
    print("\n" + "="*60)
    print("Starting Transfer Learning Training")
    print("="*60)
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            inputs = batch["input"].to(config.device)
            targets = batch["target"].to(config.device)
            
            optimizer.zero_grad()
            outputs = torch.clamp(model(inputs), 0, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_ssim, val_psnr = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(config.device)
                targets = batch["target"].to(config.device)
                outputs = torch.clamp(model(inputs), 0, 1)
                
                from skimage.metrics import structural_similarity, peak_signal_noise_ratio
                for i in range(outputs.shape[0]):
                    pred = outputs[i, 0].cpu().numpy()
                    tgt = targets[i, 0].cpu().numpy()
                    val_ssim += structural_similarity(tgt, pred, data_range=1.0)
                    val_psnr += peak_signal_noise_ratio(tgt, pred, data_range=1.0)
        
        val_ssim /= len(val_ds)
        val_psnr /= len(val_ds)
        scheduler.step()
        
        print(f"  Loss: {train_loss/len(train_loader):.4f}, SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.2f}")
        
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), config.output_dir / "best_model.pth")
            print(f"  → New best! Saved.")
        
        history.append({"epoch": epoch+1, "val_ssim": val_ssim, "val_psnr": val_psnr})
    
    # Save results
    with open(config.output_dir / "metrics.json", "w") as f:
        json.dump({
            "best_ssim": best_ssim,
            "transfer_learning": config.use_transfer,
            "encoder_lr": config.encoder_lr,
            "decoder_lr": config.decoder_lr,
            "history": history
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best SSIM: {best_ssim:.4f}")
    print(f"{'='*60}")
    
    return model, best_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--no-transfer", action="store_true", help="Train from scratch")
    parser.add_argument("--pretrained", type=str, default="notebooks/best_model.pth")
    args = parser.parse_args()
    
    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.encoder_lr = args.encoder_lr
    config.decoder_lr = args.decoder_lr
    config.use_transfer = not args.no_transfer
    config.pretrained_path = Path(args.pretrained)
    
    train_with_transfer(config)


if __name__ == "__main__":
    main()
