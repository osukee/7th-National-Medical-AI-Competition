"""
Debug script to verify model predictions locally.
This will check:
1. Model outputs valid predictions
2. Predictions are in correct range (0-1 or 0-255)
3. Compare with target to see SSIM
"""
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from pathlib import Path


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


def ssim_simple(img1, img2, data_range=255):
    """Simple SSIM calculation"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_val


def main():
    data_dir = Path("medical-ai-contest-7th-2025")
    model_path = Path("outputs/best_model.pth")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = SimpleUNet(1, 1)
    
    # Try loading state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully (SimpleUNet)")
    except Exception as e:
        print(f"Failed to load as SimpleUNet: {e}")
        print("Trying to load with SMP...")
        try:
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=1,
                classes=1,
            )
            model.load_state_dict(state_dict)
            print("Model loaded successfully (SMP Unet)")
        except Exception as e2:
            print(f"Failed to load as SMP Unet: {e2}")
            return
    
    model.to(device)
    model.eval()
    
    # Load a training sample
    sample_id = "train_00000"
    input_path = data_dir / f"train/{sample_id}.png"
    target_path = data_dir / f"train/{sample_id}_target.png"
    mask_path = data_dir / f"train/{sample_id}_mask.png"
    
    # Load and resize input
    input_img = Image.open(input_path).convert('L')
    input_img = input_img.resize((512, 512), Image.BILINEAR)
    input_arr = np.array(input_img, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(input_arr).unsqueeze(0).unsqueeze(0).to(device)
    
    # Load and resize target
    target_img = Image.open(target_path).convert('L')
    target_img = target_img.resize((512, 512), Image.BILINEAR)
    target_arr = np.array(target_img)
    
    # Load and resize mask
    mask_img = Image.open(mask_path).convert('L')
    mask_img = mask_img.resize((512, 512), Image.BILINEAR)
    mask_arr = np.array(mask_img)
    
    print(f"\nInput - min: {input_arr.min():.3f}, max: {input_arr.max():.3f}, mean: {input_arr.mean():.3f}")
    
    # Run prediction
    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.clamp(pred, 0, 1)
    
    pred_arr = pred[0, 0].cpu().numpy()
    print(f"Prediction (0-1) - min: {pred_arr.min():.3f}, max: {pred_arr.max():.3f}, mean: {pred_arr.mean():.3f}")
    
    # Convert to uint8
    pred_uint8 = (pred_arr * 255).astype(np.uint8)
    print(f"Prediction (uint8) - min: {pred_uint8.min()}, max: {pred_uint8.max()}, mean: {pred_uint8.mean():.1f}")
    
    # Calculate SSIM
    ssim_full = ssim_simple(target_arr, pred_uint8)
    print(f"\nSSIM (full image): {ssim_full:.4f}")
    
    # Calculate SSIM with mask (only on mask regions)
    mask_binary = (mask_arr > 127)
    if mask_binary.sum() > 0:
        target_masked = target_arr[mask_binary]
        pred_masked = pred_uint8[mask_binary]
        
        # Simplified SSIM on masked region (just as comparison)
        ssim_masked = ssim_simple(target_masked, pred_masked)
        print(f"SSIM (masked region only): {ssim_masked:.4f}")
    
    # Save prediction for visual inspection
    pred_img = Image.fromarray(pred_uint8)
    pred_img.save("outputs/debug_prediction.png")
    print(f"\nPrediction saved to outputs/debug_prediction.png")
    
    # Compare first 10 pixels (for submission format verification)
    flat_c = pred_uint8.flatten()
    flat_f = pred_uint8.flatten('F')
    print(f"\nFirst 10 pixels (Row-major 'C'): {flat_c[:10]}")
    print(f"First 10 pixels (Col-major 'F'): {flat_f[:10]}")


if __name__ == "__main__":
    main()
