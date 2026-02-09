"""
Fine-tune with SimCLR Pretrained Encoder

Phase 2: Use self-supervised pretrained encoder for fluorescence prediction.

ルール適合:
- Encoder: 自己教師あり事前学習済み (SimCLR)
- Decoder: 新規学習
- Loss: MSE + SSIM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


# ==============================================================================
# Configuration
# ==============================================================================
class FinetuneConfig:
    # Data
    data_dir = Path("/kaggle/input/medical-ai-contest-7th-2025")
    train_csv = data_dir / "train.csv"
    output_dir = Path("/kaggle/working")
    
    # SimCLR pretrained encoder
    simclr_encoder_path = Path("/kaggle/working/simclr_pretrained/simclr_encoder.pth")
    
    # Image
    image_size = 512
    in_channels = 1
    out_channels = 1
    
    # Training
    epochs = 30
    batch_size = 8
    encoder_lr = 1e-5    # Low LR for pretrained encoder
    decoder_lr = 1e-4    # Normal LR for decoder
    weight_decay = 1e-5
    
    # Loss
    mse_weight = 1.0
    ssim_weight = 1.0
    
    # Encoder
    encoder_name = "efficientnet-b4"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


# ==============================================================================
# Dataset
# ==============================================================================
class OrganoidDataset(Dataset):
    def __init__(self, csv_path, data_dir, image_size=512, indices=None):
        self.df = pd.read_csv(csv_path)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.image_size = image_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load input
        input_path = self.data_dir / row['input_path']
        input_img = Image.open(input_path).convert('L')
        input_img = input_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        input_arr = np.array(input_img, dtype=np.float32) / 255.0
        
        # Load target
        target_path = self.data_dir / row['target_path']
        target_img = Image.open(target_path).convert('L')
        target_img = target_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        target_arr = np.array(target_img, dtype=np.float32) / 255.0
        
        input_tensor = torch.from_numpy(input_arr).unsqueeze(0)
        target_tensor = torch.from_numpy(target_arr).unsqueeze(0)
        
        return {"id": row["id"], "input": input_tensor, "target": target_tensor}


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
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
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
# Model with SimCLR Encoder
# ==============================================================================
def create_model_with_simclr(config):
    """
    Create U-Net with SimCLR pretrained encoder.
    
    - Encoder: Loaded from SimCLR checkpoint
    - Decoder: Fresh initialization
    """
    if not SMP_AVAILABLE:
        raise ImportError("segmentation-models-pytorch required")
    
    # Create U-Net
    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=None,  # Will load from SimCLR
        in_channels=config.in_channels,
        classes=config.out_channels,
        activation='sigmoid',
    )
    
    # Load SimCLR pretrained encoder
    if config.simclr_encoder_path.exists():
        print(f"Loading SimCLR encoder from: {config.simclr_encoder_path}")
        simclr_weights = torch.load(config.simclr_encoder_path, map_location='cpu')
        
        current_state = model.encoder.state_dict()
        loaded_count = 0
        
        for key, value in simclr_weights.items():
            if key in current_state and current_state[key].shape == value.shape:
                current_state[key] = value
                loaded_count += 1
        
        model.encoder.load_state_dict(current_state)
        print(f"Loaded {loaded_count} encoder layers from SimCLR")
        
        # Reset BatchNorm
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.reset_running_stats()
    else:
        print("⚠️ SimCLR encoder not found, using random initialization")
    
    return model


def create_differential_optimizer(model, config):
    """
    Create optimizer with different learning rates.
    
    - Encoder (pretrained): Low LR
    - Decoder (fresh): Normal LR
    """
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": config.encoder_lr},
        {"params": decoder_params, "lr": config.decoder_lr},
    ], weight_decay=config.weight_decay)
    
    print(f"Optimizer: Encoder LR={config.encoder_lr}, Decoder LR={config.decoder_lr}")
    
    return optimizer


# ==============================================================================
# Training
# ==============================================================================
def train(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Fine-tuning with SimCLR Pretrained Encoder")
    print("="*60)
    
    # Data
    full_df = pd.read_csv(config.train_csv)
    n_train = int(len(full_df) * 0.8)
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(full_df)))
    
    train_dataset = OrganoidDataset(config.train_csv, config.data_dir, config.image_size, train_indices)
    val_dataset = OrganoidDataset(config.train_csv, config.data_dir, config.image_size, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = create_model_with_simclr(config)
    model = model.to(config.device)
    
    # Optimizer & Loss
    optimizer = create_differential_optimizer(model, config)
    criterion = CombinedLoss(config.mse_weight, config.ssim_weight)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    
    # Training loop
    best_ssim = 0
    history = []
    
    for epoch in range(config.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            inputs = batch["input"].to(config.device)
            targets = batch["target"].to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_ssim = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(config.device)
                targets = batch["target"].to(config.device)
                outputs = model(inputs)
                
                from skimage.metrics import structural_similarity
                for i in range(outputs.shape[0]):
                    pred = outputs[i, 0].cpu().numpy()
                    tgt = targets[i, 0].cpu().numpy()
                    val_ssim += structural_similarity(tgt, pred, data_range=1.0)
        
        val_ssim /= len(val_dataset)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val SSIM={val_ssim:.4f}")
        
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), config.output_dir / "best_model.pth")
            print(f"  → Saved best model (SSIM: {best_ssim:.4f})")
        
        history.append({"epoch": epoch+1, "val_ssim": val_ssim})
    
    # Save results
    with open(config.output_dir / "finetune_history.json", "w") as f:
        json.dump({"best_ssim": best_ssim, "history": history}, f, indent=2)
    
    print(f"\nBest Val SSIM: {best_ssim:.4f}")
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--simclr-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    config = FinetuneConfig()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
        config.train_csv = config.data_dir / "train.csv"
    if args.simclr_path:
        config.simclr_encoder_path = Path(args.simclr_path)
    config.epochs = args.epochs
    
    train(config)
