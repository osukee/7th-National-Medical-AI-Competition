# VirtualStaining Training Notebook for Google Colab
# GPU training with MSE+SSIM Loss + CLAHE preprocessing

"""
VirtualStaining Training on Google Colab

Usage:
1. Upload this file to Google Colab
2. Change runtime to GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells
"""

# ==============================================================================
# Cell 1: Install Dependencies
# ==============================================================================
# !pip install segmentation-models-pytorch opencv-python-headless scikit-image tqdm

# ==============================================================================
# Cell 2: Download Dataset from Figshare
# ==============================================================================
"""
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_figshare_dataset():
    FIGSHARE_API = "https://api.figshare.com/v2"
    ARTICLE_ID = "21971558"
    
    print("Fetching file info from Figshare...")
    response = requests.get(f"{FIGSHARE_API}/articles/{ARTICLE_ID}/files")
    files = response.json()
    
    output_dir = Path("data/VirtualStaining")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for f in files:
        print(f"Downloading {f['name']} ({f['size']/1024/1024:.1f} MB)...")
        r = requests.get(f['download_url'], stream=True)
        file_path = output_dir / f['name']
        
        with open(file_path, 'wb') as out:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                out.write(chunk)
        
        if f['name'].endswith('.zip'):
            print(f"Extracting {f['name']}...")
            with zipfile.ZipFile(file_path, 'r') as z:
                z.extractall(output_dir)
    
    print("Download complete!")

download_figshare_dataset()
"""

# ==============================================================================
# Cell 3: Imports and Configuration
# ==============================================================================
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

try:
    import segmentation_models_pytorch as smp
    USE_SMP = True
except ImportError:
    USE_SMP = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Config
DATA_DIR = Path("data/VirtualStaining/2022-09-06 Dataset Update")
IMAGE_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
MSE_WEIGHT = 1.0
SSIM_WEIGHT = 1.0
USE_CLAHE = True
"""

# ==============================================================================
# Cell 4: CLAHE Preprocessing
# ==============================================================================
"""
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if len(image.shape) == 3:
        image = image[:, :, 0]
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)
"""

# ==============================================================================
# Cell 5: Dataset
# ==============================================================================
"""
class VirtualStainingDataset(Dataset):
    def __init__(self, data_dir, image_size=512, use_clahe=True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.pairs = self._find_pairs()
        print(f"Found {len(self.pairs)} image pairs")
    
    def _find_pairs(self):
        pairs = []
        all_images = list(self.data_dir.rglob("*.jpg"))
        
        triplets = {}
        for img_path in all_images:
            stem = img_path.stem
            parts = stem.rsplit('.', 1)
            if len(parts) == 2:
                base_num, channel = parts
                key = (img_path.parent, base_num)
                if key not in triplets:
                    triplets[key] = {}
                triplets[key][channel] = img_path
        
        for (parent, base_num), channels in triplets.items():
            if '0' in channels and '1' in channels:
                pairs.append((channels['0'], channels['1']))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        phase_path, fluor_path = self.pairs[idx]
        
        # Load and preprocess input
        phase_img = cv2.imread(str(phase_path), cv2.IMREAD_GRAYSCALE)
        if self.use_clahe:
            phase_img = apply_clahe(phase_img)
        phase_img = cv2.resize(phase_img, (self.image_size, self.image_size))
        phase_arr = phase_img.astype(np.float32) / 255.0
        
        # Load target
        fluor_img = cv2.imread(str(fluor_path), cv2.IMREAD_GRAYSCALE)
        fluor_img = cv2.resize(fluor_img, (self.image_size, self.image_size))
        fluor_arr = fluor_img.astype(np.float32) / 255.0
        
        return {
            "input": torch.from_numpy(phase_arr).unsqueeze(0),
            "target": torch.from_numpy(fluor_arr).unsqueeze(0),
        }
"""

# ==============================================================================
# Cell 6: Loss Functions
# ==============================================================================
"""
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2*1.5**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        _1D = gauss.unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', _2D.expand(1, 1, window_size, window_size).contiguous())
    
    def forward(self, img1, img2):
        mu1 = nn.functional.conv2d(img1, self.window, padding=self.window_size//2, groups=1)
        mu2 = nn.functional.conv2d(img2, self.window, padding=self.window_size//2, groups=1)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        
        sigma1_sq = nn.functional.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=1) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=1) - mu2_sq
        sigma12 = nn.functional.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=1) - mu1_mu2
        
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class MSESSIMLoss(nn.Module):
    def __init__(self, mse_weight=1.0, ssim_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + self.ssim_weight * self.ssim(pred, target)
"""

# ==============================================================================
# Cell 7: Model
# ==============================================================================
"""
if USE_SMP:
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
else:
    # Simple U-Net fallback
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.conv(x)
    
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1, self.enc2, self.enc3, self.enc4 = DoubleConv(1, 64), DoubleConv(64, 128), DoubleConv(128, 256), DoubleConv(256, 512)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(512, 1024)
            self.up4, self.dec4 = nn.ConvTranspose2d(1024, 512, 2, stride=2), DoubleConv(1024, 512)
            self.up3, self.dec3 = nn.ConvTranspose2d(512, 256, 2, stride=2), DoubleConv(512, 256)
            self.up2, self.dec2 = nn.ConvTranspose2d(256, 128, 2, stride=2), DoubleConv(256, 128)
            self.up1, self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64)
            self.out_conv = nn.Conv2d(64, 1, 1)
        
        def forward(self, x):
            e1, e2 = self.enc1(x), self.enc2(self.pool(self.enc1(x)))
            e3, e4 = self.enc3(self.pool(e2)), self.enc4(self.pool(self.enc3(self.pool(e2))))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
            return self.out_conv(d1)
    
    model = SimpleUNet()

model = model.to(device)
print(f"Model: {'EfficientNet-B4 U-Net' if USE_SMP else 'Simple U-Net'}")
"""

# ==============================================================================
# Cell 8: Training
# ==============================================================================
"""
# Dataset & DataLoader
dataset = VirtualStainingDataset(DATA_DIR, IMAGE_SIZE, USE_CLAHE)
n_train = int(len(dataset) * 0.8)
train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Training setup
criterion = MSESSIMLoss(MSE_WEIGHT, SSIM_WEIGHT).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

best_ssim = 0
history = []

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        
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
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            outputs = torch.clamp(model(inputs), 0, 1)
            
            # Calculate metrics
            pred_np = outputs.cpu().numpy()
            target_np = targets.cpu().numpy()
            from skimage.metrics import structural_similarity, peak_signal_noise_ratio
            for i in range(pred_np.shape[0]):
                val_ssim += structural_similarity(target_np[i,0], pred_np[i,0], data_range=1.0)
                val_psnr += peak_signal_noise_ratio(target_np[i,0], pred_np[i,0], data_range=1.0)
    
    val_ssim /= len(val_ds)
    val_psnr /= len(val_ds)
    scheduler.step()
    
    print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val SSIM: {val_ssim:.4f}, Val PSNR: {val_psnr:.2f}")
    
    if val_ssim > best_ssim:
        best_ssim = val_ssim
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  → Saved best model!")
    
    history.append({"epoch": epoch+1, "val_ssim": val_ssim, "val_psnr": val_psnr})

print(f"\\n=== Training Complete ===")
print(f"Best SSIM: {best_ssim:.4f}")

# Save metrics
with open("training_metrics.json", "w") as f:
    json.dump({"best_ssim": best_ssim, "history": history}, f, indent=2)
"""

# ==============================================================================
# Cell 9: Download trained model
# ==============================================================================
"""
from google.colab import files
files.download("best_model.pth")
files.download("training_metrics.json")
"""
