"""
Quick Local Experiment Script
ローカルで小さな仮説を素早く検証するためのスクリプト

使い方:
    python scripts/quick_experiment.py --epochs 3 --samples 20 --save-samples

方針:
    1. 小さな仮説を立てる
    2. 1変数だけ変える
    3. ログと失敗例を"目で見る"
"""

import argparse
import json
import os
import sys
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
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import segmentation_models_pytorch as smp
    USE_SMP = True
except ImportError:
    USE_SMP = False
    print("Warning: smp not found, using simple model")


class QuickDataset(Dataset):
    """シンプルなデータセット（高速読み込み用）"""
    def __init__(self, csv_path, data_dir, image_size=256):
        self.df = pd.read_csv(csv_path)
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Input
        input_path = self.data_dir / row["input_path"]
        input_img = Image.open(input_path).convert("L")
        input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        input_arr = np.array(input_img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(input_arr).unsqueeze(0)
        
        # Target
        target_path = self.data_dir / row["target_path"]
        target_img = Image.open(target_path).convert("L")
        target_img = target_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        target_arr = np.array(target_img, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(target_arr).unsqueeze(0)
        
        return {
            "id": row["id"],
            "input": input_tensor,
            "target": target_tensor,
        }


class SimpleUNet(nn.Module):
    """シンプルなU-Net（高速実験用）"""
    def __init__(self, channels=[32, 64, 128]):
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(1, channels[0])
        self.enc2 = self._block(channels[0], channels[1])
        self.enc3 = self._block(channels[1], channels[2])
        
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = self._block(channels[1]*2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = self._block(channels[0]*2, channels[0])
        
        self.out = nn.Conv2d(channels[0], 1, 1)
        
    def _block(self, in_ch, out_ch):
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
        
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.sigmoid(self.out(d1))


def calculate_metrics(pred, target):
    """バッチのメトリクス計算"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    ssim_scores = []
    psnr_scores = []
    
    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i, 0], 0, 1)
        t = np.clip(target_np[i, 0], 0, 1)
        
        ssim_val = ssim(t, p, data_range=1.0)
        psnr_val = psnr(t, p, data_range=1.0)
        
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)
    
    return np.mean(ssim_scores), np.mean(psnr_scores)


def save_comparison_image(inputs, targets, preds, save_path, epoch):
    """比較画像を保存（目で見る用）"""
    n = min(4, inputs.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        inp = inputs[i, 0].cpu().numpy()
        tgt = targets[i, 0].cpu().numpy()
        pred = preds[i, 0].detach().cpu().numpy()
        
        # 差分計算
        diff = np.abs(pred - tgt)
        
        axes[i, 0].imshow(inp, cmap='gray')
        axes[i, 0].set_title(f'Input')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(tgt, cmap='gray')
        axes[i, 1].set_title(f'Target')
        axes[i, 1].axis('off')
        
        # 予測と差分をオーバーレイ
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].imshow(diff, cmap='Reds', alpha=0.3)
        ssim_val = ssim(tgt, np.clip(pred, 0, 1), data_range=1.0)
        axes[i, 2].set_title(f'Pred (SSIM: {ssim_val:.3f})')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def run_experiment(args):
    """実験実行"""
    print("=" * 60)
    print(f"Quick Experiment: {args.exp_name}")
    print(f"  Epochs: {args.epochs}, Samples: {args.samples}, Image Size: {args.image_size}")
    print("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Data
    data_dir = Path("medical-ai-contest-7th-2025")
    dataset = QuickDataset(data_dir / "train.csv", data_dir, args.image_size)
    
    # サンプル数制限
    indices = list(range(min(args.samples, len(dataset))))
    train_indices = indices[:int(len(indices)*0.8)]
    val_indices = indices[int(len(indices)*0.8):]
    
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    
    # Model
    if USE_SMP and args.use_smp:
        model = smp.Unet(
            encoder_name="resnet18",  # 軽量エンコーダ
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )
    else:
        model = SimpleUNet()
    
    model = model.to(device)
    print(f"Model: {type(model).__name__}")
    
    # Loss & Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Output directory
    output_dir = Path("outputs") / "quick_experiments" / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    history = []
    best_ssim = 0
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_ssim, val_psnr = 0, 0
        n_val = 0
        last_batch = None
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                outputs = model(inputs)
                
                batch_ssim, batch_psnr = calculate_metrics(outputs, targets)
                val_ssim += batch_ssim
                val_psnr += batch_psnr
                n_val += 1
                last_batch = (inputs, targets, outputs)
        
        val_ssim /= n_val
        val_psnr /= n_val
        
        # Log
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, SSIM={val_ssim:.4f}, PSNR={val_psnr:.2f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_ssim": val_ssim,
            "val_psnr": val_psnr,
        })
        
        # Save comparison images
        if args.save_samples and last_batch is not None:
            save_comparison_image(
                last_batch[0], last_batch[1], last_batch[2],
                output_dir / f"epoch_{epoch+1:02d}.png",
                epoch + 1
            )
        
        # Best model
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), output_dir / "best_model.pth")
    
    # Save results
    results = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "best_ssim": float(best_ssim),
        "final_psnr": float(history[-1]["val_psnr"]),
        "config": vars(args),
        "history": history,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results: SSIM={best_ssim:.4f}")
    print(f"Saved to: {output_dir}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Quick Local Experiment")
    parser.add_argument("--exp-name", type=str, default=f"exp_{datetime.now().strftime('%H%M%S')}")
    parser.add_argument("--epochs", type=int, default=3, help="少数エポックで素早く確認")
    parser.add_argument("--samples", type=int, default=50, help="使用サンプル数")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256, help="小さい画像で高速化")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-smp", action="store_true", help="SMP使用（重い）")
    parser.add_argument("--save-samples", action="store_true", help="比較画像を保存")
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
