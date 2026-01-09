"""
Mask-based evaluation utilities for Medical AI Competition.
Matches competition scoring: Score = (SSIM + Normalized_PSNR) / 2
"""
import numpy as np
from PIL import Image
from pathlib import Path


def calculate_ssim_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, 
                          data_range: int = 255) -> float:
    """
    Calculate SSIM only on masked region.
    
    Args:
        pred: Predicted image (H, W), uint8 or float
        target: Ground truth image (H, W), uint8 or float
        mask: Binary mask (H, W), >0 = evaluate
        data_range: Max value (255 for uint8)
    
    Returns:
        SSIM value for masked region
    """
    # Convert to float64
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mask_bool = mask > 0
    
    if mask_bool.sum() == 0:
        return 0.0
    
    # Extract masked pixels
    pred_m = pred[mask_bool]
    target_m = target[mask_bool]
    
    # SSIM constants
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


def calculate_psnr_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                          data_range: int = 255) -> float:
    """
    Calculate PSNR only on masked region.
    
    Args:
        pred: Predicted image (H, W), uint8 or float
        target: Ground truth image (H, W), uint8 or float
        mask: Binary mask (H, W), >0 = evaluate
        data_range: Max value (255 for uint8)
    
    Returns:
        PSNR value in dB for masked region
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
        return 100.0  # Perfect match
    
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def normalize_psnr(psnr: float, min_psnr: float = 15.0, max_psnr: float = 35.0) -> float:
    """
    Normalize PSNR to 0-1 range for competition scoring.
    
    Args:
        psnr: Raw PSNR value in dB
        min_psnr: PSNR value that maps to 0 (default: 15 dB)
        max_psnr: PSNR value that maps to 1 (default: 35 dB)
    
    Returns:
        Normalized PSNR (clipped to 0-1)
    """
    normalized = (psnr - min_psnr) / (max_psnr - min_psnr)
    return float(np.clip(normalized, 0.0, 1.0))


def calculate_competition_score(pred: np.ndarray, target: np.ndarray, 
                                mask: np.ndarray) -> dict:
    """
    Calculate competition score using mask-based evaluation.
    
    Formula: Score = (SSIM + Normalized_PSNR) / 2
    
    Returns:
        Dictionary with ssim, psnr, psnr_normalized, score
    """
    ssim = calculate_ssim_masked(pred, target, mask)
    psnr = calculate_psnr_masked(pred, target, mask)
    psnr_norm = normalize_psnr(psnr)
    score = (ssim + psnr_norm) / 2
    
    return {
        'ssim': ssim,
        'psnr': psnr,
        'psnr_normalized': psnr_norm,
        'score': score,
    }


def evaluate_sample(pred_path: str, target_path: str, mask_path: str,
                    resize_to: tuple = (512, 512)) -> dict:
    """
    Evaluate a single sample.
    
    Args:
        pred_path: Path to predicted image
        target_path: Path to target image
        mask_path: Path to mask image
        resize_to: Resize all images to this size
    
    Returns:
        Metrics dictionary
    """
    # Load images
    pred = Image.open(pred_path).convert('L')
    target = Image.open(target_path).convert('L')
    mask = Image.open(mask_path).convert('L')
    
    # Resize
    if resize_to:
        pred = pred.resize(resize_to, Image.BILINEAR)
        target = target.resize(resize_to, Image.BILINEAR)
        mask = mask.resize(resize_to, Image.NEAREST)  # NEAREST for mask!
    
    # Convert to numpy
    pred_arr = np.array(pred)
    target_arr = np.array(target)
    mask_arr = np.array(mask)
    
    # Binarize mask (threshold at 127)
    mask_arr = (mask_arr > 127).astype(np.uint8) * 255
    
    return calculate_competition_score(pred_arr, target_arr, mask_arr)


def main():
    """Test evaluation on training samples."""
    data_dir = Path("medical-ai-contest-7th-2025")
    
    print("=" * 60)
    print("Mask-Based Evaluation Test")
    print("=" * 60)
    
    # Test on first 5 training samples (using input as "prediction" for baseline)
    results = []
    
    for i in range(5):
        sample_id = f"train_{i:05d}"
        input_path = data_dir / f"train/{sample_id}.png"
        target_path = data_dir / f"train/{sample_id}_target.png"
        mask_path = data_dir / f"train/{sample_id}_mask.png"
        
        if not all(p.exists() for p in [input_path, target_path, mask_path]):
            continue
        
        # Load images
        input_img = np.array(Image.open(input_path).convert('L').resize((512, 512)))
        target_img = np.array(Image.open(target_path).convert('L').resize((512, 512)))
        mask_img = np.array(Image.open(mask_path).convert('L').resize((512, 512), Image.NEAREST))
        mask_img = (mask_img > 127).astype(np.uint8) * 255
        
        # Calculate metrics using INPUT as prediction (baseline)
        metrics = calculate_competition_score(input_img, target_img, mask_img)
        results.append(metrics)
        
        print(f"\n{sample_id}:")
        print(f"  SSIM (mask): {metrics['ssim']:.4f}")
        print(f"  PSNR (mask): {metrics['psnr']:.2f} dB")
        print(f"  PSNR (norm): {metrics['psnr_normalized']:.4f}")
        print(f"  Score:       {metrics['score']:.4f}")
    
    # Aggregate
    if results:
        print("\n" + "=" * 60)
        print("Aggregate (Input as prediction - baseline):")
        print(f"  SSIM mean:  {np.mean([r['ssim'] for r in results]):.4f}")
        print(f"  PSNR mean:  {np.mean([r['psnr'] for r in results]):.2f} dB")
        print(f"  Score mean: {np.mean([r['score'] for r in results]):.4f}")
        print("=" * 60)
        
    # Compare with full-image evaluation
    print("\n" + "=" * 60)
    print("Comparison: Full Image vs Mask Only")
    print("=" * 60)
    
    for i in range(2):
        sample_id = f"train_{i:05d}"
        input_img = np.array(Image.open(data_dir / f"train/{sample_id}.png").convert('L').resize((512, 512)))
        target_img = np.array(Image.open(data_dir / f"train/{sample_id}_target.png").convert('L').resize((512, 512)))
        mask_img = np.array(Image.open(data_dir / f"train/{sample_id}_mask.png").convert('L').resize((512, 512), Image.NEAREST))
        mask_img = (mask_img > 127).astype(np.uint8) * 255
        
        # Full image
        full_mask = np.ones_like(mask_img) * 255
        full_metrics = calculate_competition_score(input_img, target_img, full_mask)
        
        # Mask only
        mask_metrics = calculate_competition_score(input_img, target_img, mask_img)
        
        print(f"\n{sample_id}:")
        print(f"  Full Image - SSIM: {full_metrics['ssim']:.4f}, PSNR: {full_metrics['psnr']:.2f}")
        print(f"  Mask Only  - SSIM: {mask_metrics['ssim']:.4f}, PSNR: {mask_metrics['psnr']:.2f}")


if __name__ == "__main__":
    main()
