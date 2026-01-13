"""
Test script to verify hypothesis about flatten order (Row-major vs Column-major)
"""
import numpy as np
from PIL import Image


def calculate_ssim(img1, img2, data_range=255):
    """Simple SSIM implementation (window-based method simplified)"""
    # Normalize
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
    # Load a training target image
    target_path = 'medical-ai-contest-7th-2025/train/train_00000_target.png'
    target = Image.open(target_path).convert('L')
    print(f"Original target size: {target.size}")
    
    # Resize to 512x512 (as submission format requires)
    target_512 = target.resize((512, 512), Image.BILINEAR)
    target_arr = np.array(target_512)
    print(f"Resized target shape: {target_arr.shape}")
    
    # Test 1: Flatten with Row-major ('C'), reconstruct with Row-major
    flat_c = target_arr.flatten()  # What we currently submit
    reconstructed_c_c = flat_c.reshape(512, 512)  # If eval uses C order
    ssim_c_c = calculate_ssim(target_arr, reconstructed_c_c)
    print(f"\n[Test 1] Flatten C, Reconstruct C: SSIM = {ssim_c_c:.4f}")
    
    # Test 2: Flatten with Row-major ('C'), reconstruct with Column-major ('F')
    reconstructed_c_f = flat_c.reshape(512, 512, order='F')  # If eval uses F order
    ssim_c_f = calculate_ssim(target_arr, reconstructed_c_f)
    print(f"[Test 2] Flatten C, Reconstruct F: SSIM = {ssim_c_f:.4f}")
    
    # Test 3: Flatten with Column-major ('F'), reconstruct with Column-major
    flat_f = target_arr.flatten('F')  # Alternative submit format
    reconstructed_f_f = flat_f.reshape(512, 512, order='F')
    ssim_f_f = calculate_ssim(target_arr, reconstructed_f_f)
    print(f"[Test 3] Flatten F, Reconstruct F: SSIM = {ssim_f_f:.4f}")
    
    # Test 4: Flatten with Column-major ('F'), reconstruct with Row-major ('C')
    reconstructed_f_c = flat_f.reshape(512, 512)
    ssim_f_c = calculate_ssim(target_arr, reconstructed_f_c)
    print(f"[Test 4] Flatten F, Reconstruct C: SSIM = {ssim_f_c:.4f}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS:")
    if ssim_c_f < 0.5:
        print(f"  ⚠️  If we submit Row-major but eval uses Column-major:")
        print(f"      SSIM drops to {ssim_c_f:.4f} - matches PB score ~0.35!")
        print(f"  → TRY: Change flatten() to flatten('F')")
    else:
        print(f"  Order mismatch does NOT explain the issue.")
        print(f"  Look for other causes.")
    print("="*60)


if __name__ == "__main__":
    main()
