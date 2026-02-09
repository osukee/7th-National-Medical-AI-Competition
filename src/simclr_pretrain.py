"""
SimCLR Self-Supervised Pretraining Script

ルール適合：本番入力画像のみ使用（ラベル不使用）

Phase 1: Contrastive learningでEncoderを事前学習
Phase 2: Fine-tuneで蛍光予測

Reference: "A Simple Framework for Contrastive Learning of Visual Representations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json

# Try to import SMP for encoder
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: SMP not available, using simple encoder")


# ==============================================================================
# Configuration
# ==============================================================================
class SimCLRConfig:
    # Data
    data_dir = Path("/kaggle/input/medical-ai-contest-7th-2025")
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    output_dir = Path("/kaggle/working/simclr_pretrained")
    
    # Image
    image_size = 512
    
    # SimCLR
    temperature = 0.5  # NT-Xent temperature
    projection_dim = 128  # Projection head output dimension
    
    # Training
    epochs = 100
    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 1e-4
    
    # Encoder
    encoder_name = "efficientnet-b4"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


# ==============================================================================
# SimCLR Augmentations
# ==============================================================================
class SimCLRAugmentation:
    """
    SimCLR-style augmentations for medical images.
    
    Key insight: Strong augmentation creates diverse views for contrastive learning.
    """
    def __init__(self, image_size=512):
        self.image_size = image_size
    
    def __call__(self, img_arr):
        """Apply augmentation to create a view."""
        import cv2
        
        # Random crop and resize
        h, w = img_arr.shape[:2]
        scale = np.random.uniform(0.8, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        img = img_arr[top:top+new_h, left:left+new_w]
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Random horizontal flip (50%)
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
        
        # Random vertical flip (50%)
        if np.random.random() > 0.5:
            img = np.flipud(img).copy()
        
        # Random rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)
        img = np.rot90(img, k).copy()
        
        # Gaussian blur (50%)
        if np.random.random() > 0.5:
            ksize = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        # Brightness/contrast adjustment
        alpha = np.random.uniform(0.8, 1.2)  # contrast
        beta = np.random.uniform(-0.1, 0.1)  # brightness
        img = np.clip(alpha * img + beta, 0, 1)
        
        return img.astype(np.float32)


# ==============================================================================
# Dataset (Input images only - NO LABELS)
# ==============================================================================
class SimCLRDataset(Dataset):
    """
    Dataset for SimCLR pretraining.
    
    IMPORTANT: Uses only input images (phase-contrast).
    Does NOT use target/label images (fluorescence).
    """
    def __init__(self, data_dir, image_size=512):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = SimCLRAugmentation(image_size)
        
        # Collect all input images (train + test)
        self.image_paths = []
        
        # Train images
        train_csv = self.data_dir / "train.csv"
        if train_csv.exists():
            train_df = pd.read_csv(train_csv)
            for _, row in train_df.iterrows():
                self.image_paths.append(self.data_dir / row['input_path'])
        
        # Test images (also usable for SSL - no labels needed)
        test_csv = self.data_dir / "test.csv"
        if test_csv.exists():
            test_df = pd.read_csv(test_csv)
            for _, row in test_df.iterrows():
                self.image_paths.append(self.data_dir / row['input_path'])
        
        print(f"SimCLR Dataset: {len(self.image_paths)} images (train + test)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        
        # Create two augmented views
        view1 = self.augment(img_arr)
        view2 = self.augment(img_arr)
        
        # Convert to tensors (add channel dimension)
        view1 = torch.from_numpy(view1).unsqueeze(0)
        view2 = torch.from_numpy(view2).unsqueeze(0)
        
        return view1, view2


# ==============================================================================
# SimCLR Model
# ==============================================================================
class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR."""
    def __init__(self, in_features, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class SimCLREncoder(nn.Module):
    """
    SimCLR encoder with projection head.
    
    Uses EfficientNet encoder from SMP for feature extraction.
    """
    def __init__(self, encoder_name="efficientnet-b4", projection_dim=128):
        super().__init__()
        
        if SMP_AVAILABLE:
            # Use SMP encoder (same as will be used in fine-tuning)
            self.encoder = smp.encoders.get_encoder(
                name=encoder_name,
                in_channels=1,
                weights="imagenet" if encoder_name.startswith("efficientnet") else None,
            )
            # Get encoder output dimension
            encoder_out_dim = self.encoder.out_channels[-1]
        else:
            # Fallback: simple CNN encoder
            self.encoder = self._create_simple_encoder()
            encoder_out_dim = 512
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection head
        self.projection = ProjectionHead(encoder_out_dim, out_dim=projection_dim)
    
    def _create_simple_encoder(self):
        """Simple CNN encoder as fallback."""
        return nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        if SMP_AVAILABLE:
            features = self.encoder(x)[-1]  # Get last feature map
        else:
            features = self.encoder(x)
        
        # Global average pooling
        features = self.pool(features).flatten(1)
        
        # Projection
        z = self.projection(features)
        
        return F.normalize(z, dim=1)  # L2 normalize
    
    def get_encoder_state_dict(self):
        """Get only encoder weights (for fine-tuning)."""
        return self.encoder.state_dict()


# ==============================================================================
# NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)
# ==============================================================================
class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning.
    
    For each positive pair (i, j):
    - Numerator: exp(sim(z_i, z_j) / temperature)
    - Denominator: sum of exp(sim(z_i, z_k) / temperature) for all k != i
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Args:
            z1: (N, D) normalized embeddings from view 1
            z2: (N, D) normalized embeddings from view 2
        """
        batch_size = z1.shape[0]
        
        # Concatenate embeddings: [z1_0, z1_1, ..., z2_0, z2_1, ...]
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -float('inf'))
        
        # Positive pairs: (i, i+N) and (i+N, i)
        pos_mask = torch.zeros_like(sim).bool()
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True
        
        # Extract positive similarities
        pos_sim = sim[pos_mask].view(2 * batch_size, 1)
        
        # Negative similarities (all except self and positive)
        neg_mask = ~mask & ~pos_mask
        neg_sim = sim[neg_mask].view(2 * batch_size, -1)
        
        # Compute loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


# ==============================================================================
# Training
# ==============================================================================
def train_simclr(config):
    """Train SimCLR encoder."""
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SimCLR Self-Supervised Pretraining")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Encoder: {config.encoder_name}")
    print(f"Temperature: {config.temperature}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    
    # Dataset
    dataset = SimCLRDataset(config.data_dir, config.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,  # Important for contrastive learning
    )
    
    # Model
    model = SimCLREncoder(config.encoder_name, config.projection_dim)
    model = model.to(config.device)
    
    # Loss and optimizer
    criterion = NTXentLoss(config.temperature)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    
    # Training loop
    best_loss = float('inf')
    history = []
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for view1, view2 in pbar:
            view1 = view1.to(config.device)
            view2 = view2.to(config.device)
            
            # Forward
            z1 = model(view1)
            z2 = model(view2)
            
            # Loss
            loss = criterion(z1, z2)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Epoch stats
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": scheduler.get_last_lr()[0],
        })
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.get_encoder_state_dict(),
                config.output_dir / "simclr_encoder.pth"
            )
            print(f"  → Saved best encoder (loss: {best_loss:.4f})")
    
    # Save training history
    with open(config.output_dir / "simclr_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Encoder saved to: {config.output_dir / 'simclr_encoder.pth'}")
    print("="*60)
    
    return model


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()
    
    config = SimCLRConfig()
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
        config.train_csv = config.data_dir / "train.csv"
        config.test_csv = config.data_dir / "test.csv"
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.temperature = args.temperature
    
    train_simclr(config)
