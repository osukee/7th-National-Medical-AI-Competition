"""
Pipeline Debug Script for Medical AI Competition

Purpose: Validate the entire evaluation pipeline to identify potential bugs
that could be causing score discrepancies between CV and LB.

Usage:
    python scripts/debug_pipeline.py

Checks:
1. Mask verification (shape, values, coverage)
2. SSIM/PSNR calculation alignment with LB
3. CSV round-trip integrity (write → read → verify)
4. Float→uint8 conversion verification
5. Row-major flatten order verification
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage


# =============================================================================
# Configuration
# =============================================================================

# Default paths (adjust for your environment)
DATA_DIR = Path("medical-ai-contest-7th-2025")  # Relative to repo root
OUTPUT_DIR = Path("outputs/debug")


# =============================================================================
# Metric Calculation (LB-Aligned)
# =============================================================================

def calculate_ssim_masked(pred, target, mask, data_range=255):
    """
    Calculate SSIM only on masked region (LB-aligned).
    
    CRITICAL: This MUST match the LB evaluation exactly.
    
    Args:
        pred: Predicted image (H, W), uint8 [0, 255]
        target: Ground truth image (H, W), uint8 [0, 255]
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
    
    # LB uses these exact C1, C2 values
    C1 = (0.01 * data_range) ** 2  # (2.55)^2 = 6.5025
    C2 = (0.03 * data_range) ** 2  # (7.65)^2 = 58.5225
    
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
    
    Args:
        pred: Predicted image (H, W), uint8 [0, 255]
        target: Ground truth image (H, W), uint8 [0, 255]
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
    
    mse = np.mean((pred_m - target_m) ** 2)
    
    if mse == 0:
        return 100.0
    
    psnr_val = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr_val)


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


# =============================================================================
# Verification Functions
# =============================================================================

def verify_mask(mask_path, expected_shape=(512, 512)):
    """
    Verify mask file is correctly formatted.
    
    Checks:
    - Shape matches expected
    - Values are binary (0 or 255, or 0/1)
    - Coverage ratio is reasonable (10-90%)
    """
    errors = []
    warnings = []
    
    try:
        mask = np.array(Image.open(mask_path).convert('L'))
    except Exception as e:
        return {"errors": [f"Failed to load mask: {e}"], "warnings": []}
    
    # Shape check
    if mask.shape != expected_shape:
        errors.append(f"Mask shape {mask.shape} != expected {expected_shape}")
    
    # Value check
    unique_values = np.unique(mask)
    if len(unique_values) > 2:
        warnings.append(f"Mask has {len(unique_values)} unique values (expected 2): {unique_values[:10]}...")
    
    # Coverage check
    coverage = (mask > 0).sum() / mask.size
    if coverage < 0.1:
        warnings.append(f"Mask coverage is very low: {coverage:.1%}")
    elif coverage > 0.9:
        warnings.append(f"Mask coverage is very high: {coverage:.1%}")
    
    return {
        "shape": mask.shape,
        "unique_values": unique_values.tolist(),
        "coverage": coverage,
        "errors": errors,
        "warnings": warnings,
    }


def verify_csv_roundtrip(pixels_flat, sample_id, output_dir):
    """
    Verify CSV round-trip: flatten → CSV row → read back → reshape.
    
    This catches issues with:
    - Flatten order (row-major vs column-major)
    - Data type conversion
    - Value clipping
    """
    errors = []
    
    # Original shape
    expected_shape = (512, 512)
    expected_pixels = 512 * 512  # 262,144
    
    # Check flat array length
    if len(pixels_flat) != expected_pixels:
        errors.append(f"Pixel count {len(pixels_flat)} != expected {expected_pixels}")
        return {"errors": errors}
    
    # Simulate CSV write/read (as string conversion)
    csv_row = [str(int(p)) for p in pixels_flat]
    read_back = np.array([int(p) for p in csv_row], dtype=np.uint8)
    
    # Reshape (row-major order)
    reconstructed = read_back.reshape(expected_shape, order='C')
    
    # Verify original vs reconstructed
    original_2d = pixels_flat.reshape(expected_shape, order='C')
    
    if not np.array_equal(original_2d, reconstructed):
        errors.append("Round-trip mismatch!")
        diff = np.abs(original_2d.astype(np.int32) - reconstructed.astype(np.int32))
        errors.append(f"Max diff: {diff.max()}, Mean diff: {diff.mean():.4f}")
    
    # Save debug image
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(reconstructed).save(output_dir / f"{sample_id}_roundtrip.png")
    
    return {
        "shape": reconstructed.shape,
        "dtype": str(reconstructed.dtype),
        "min": int(reconstructed.min()),
        "max": int(reconstructed.max()),
        "mean": float(reconstructed.mean()),
        "errors": errors,
    }


def verify_float_to_uint8(pred_float):
    """
    Verify float→uint8 conversion is correct.
    
    Expected: pred_float in [0, 1] → pred_uint8 in [0, 255]
    
    Checks:
    - Values are clipped to [0, 255]
    - No NaN or Inf values
    - Precision loss is minimal
    """
    errors = []
    warnings = []
    
    # Check for NaN/Inf
    if np.isnan(pred_float).any():
        errors.append(f"Found {np.isnan(pred_float).sum()} NaN values!")
    if np.isinf(pred_float).any():
        errors.append(f"Found {np.isinf(pred_float).sum()} Inf values!")
    
    # Check value range
    if pred_float.min() < -0.01:
        warnings.append(f"pred_float has negative values: min={pred_float.min():.4f}")
    if pred_float.max() > 1.01:
        warnings.append(f"pred_float exceeds 1: max={pred_float.max():.4f}")
    
    # Standard conversion
    pred_uint8 = np.clip(pred_float * 255, 0, 255).astype(np.uint8)
    
    # Alternative: round before conversion
    pred_uint8_rounded = np.clip(np.round(pred_float * 255), 0, 255).astype(np.uint8)
    
    # Compare
    diff = np.abs(pred_uint8.astype(np.int32) - pred_uint8_rounded.astype(np.int32))
    if diff.max() > 0:
        warnings.append(f"Rounding makes a difference: max diff = {diff.max()}")
    
    return {
        "input_range": (float(pred_float.min()), float(pred_float.max())),
        "output_range": (int(pred_uint8.min()), int(pred_uint8.max())),
        "errors": errors,
        "warnings": warnings,
    }


def verify_flatten_order():
    """
    Verify that flatten order matches LB expectation (row-major).
    
    Creates a test image and checks that flattening produces expected sequence.
    """
    # Create test image with known pattern
    # Row 0: [0, 1, 2, ...]
    # Row 1: [512, 513, 514, ...]
    # etc.
    test_img = np.arange(512 * 512, dtype=np.uint16).reshape(512, 512)
    
    # Flatten with row-major order (C order)
    flat_c = test_img.flatten(order='C')
    
    # Flatten with column-major order (F order)
    flat_f = test_img.flatten(order='F')
    
    # Row-major should give: [0, 1, 2, ..., 511, 512, 513, ...]
    expected_first_5 = [0, 1, 2, 3, 4]
    expected_at_512 = 512  # First element of second row
    
    c_correct = (
        list(flat_c[:5]) == expected_first_5 and
        flat_c[512] == expected_at_512
    )
    
    return {
        "row_major_correct": c_correct,
        "c_order_first_5": flat_c[:5].tolist(),
        "c_order_at_512": int(flat_c[512]),
        "f_order_first_5": flat_f[:5].tolist(),
        "f_order_at_512": int(flat_f[512]),
    }


# =============================================================================
# Full Pipeline Check
# =============================================================================

def run_full_check(data_dir, sample_id=None, output_dir=None):
    """
    Run all verification checks on the pipeline.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("=" * 60)
    print("Pipeline Debug Check")
    print("=" * 60)
    
    # 1. Flatten order check
    print("\n[1] Flatten Order Check...")
    results["flatten_order"] = verify_flatten_order()
    if results["flatten_order"]["row_major_correct"]:
        print("   ✅ Row-major (C-order) flatten is correct")
    else:
        print("   ❌ Flatten order issue detected!")
    
    # 2. Load train.csv to get sample info
    train_csv = data_dir / "train.csv"
    if not train_csv.exists():
        print(f"\n❌ train.csv not found at {train_csv}")
        return results
    
    df = pd.read_csv(train_csv)
    print(f"\n[2] Loaded {len(df)} training samples")
    
    # 3. Pick a sample to test
    if sample_id is None:
        sample_id = df.iloc[0]['id']
    
    sample_row = df[df['id'] == sample_id].iloc[0]
    print(f"\n[3] Testing sample: {sample_id}")
    
    # 4. Mask verification
    if 'mask_path' in sample_row and pd.notna(sample_row.get('mask_path')):
        mask_path = data_dir / sample_row['mask_path']
        print(f"   Mask path: {mask_path}")
        results["mask"] = verify_mask(mask_path)
        
        if results["mask"]["errors"]:
            print(f"   ❌ Mask errors: {results['mask']['errors']}")
        else:
            print(f"   ✅ Mask OK: shape={results['mask']['shape']}, coverage={results['mask']['coverage']:.1%}")
        
        if results["mask"]["warnings"]:
            for w in results["mask"]["warnings"]:
                print(f"   ⚠️  {w}")
    else:
        print("   ⚠️  No mask_path in CSV")
    
    # 5. Load input and target
    input_path = data_dir / sample_row['input_path']
    target_path = data_dir / sample_row['target_path']
    
    input_img = np.array(Image.open(input_path).convert('L'))
    target_img = np.array(Image.open(target_path).convert('L'))
    
    print(f"\n[4] Image Stats:")
    print(f"   Input:  shape={input_img.shape}, dtype={input_img.dtype}, range=[{input_img.min()}, {input_img.max()}]")
    print(f"   Target: shape={target_img.shape}, dtype={target_img.dtype}, range=[{target_img.min()}, {target_img.max()}]")
    
    # 6. Float to uint8 verification
    print("\n[5] Float→Uint8 Conversion Check...")
    # Simulate model output (float 0-1)
    simulated_pred_float = target_img.astype(np.float32) / 255.0
    results["float_to_uint8"] = verify_float_to_uint8(simulated_pred_float)
    
    if results["float_to_uint8"]["errors"]:
        for e in results["float_to_uint8"]["errors"]:
            print(f"   ❌ {e}")
    else:
        print("   ✅ Float→Uint8 conversion OK")
    
    if results["float_to_uint8"]["warnings"]:
        for w in results["float_to_uint8"]["warnings"]:
            print(f"   ⚠️  {w}")
    
    # 7. CSV Round-trip verification
    print("\n[6] CSV Round-Trip Check...")
    test_pixels = target_img.flatten(order='C')
    results["csv_roundtrip"] = verify_csv_roundtrip(test_pixels, sample_id, output_dir)
    
    if results["csv_roundtrip"]["errors"]:
        for e in results["csv_roundtrip"]["errors"]:
            print(f"   ❌ {e}")
    else:
        print("   ✅ CSV round-trip OK")
    
    # 8. Metric calculation verification
    print("\n[7] Metric Calculation Check...")
    mask_path = data_dir / sample_row['mask_path']
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # Calculate with our implementation
    our_ssim = calculate_ssim_masked(target_img, target_img, mask, data_range=255)
    our_psnr = calculate_psnr_masked(target_img, target_img, mask, data_range=255)
    
    # Perfect match should give SSIM=1, PSNR=100 (or inf)
    print(f"   Self-comparison (should be perfect):")
    print(f"   SSIM = {our_ssim:.6f} (expected: 1.0)")
    print(f"   PSNR = {our_psnr:.2f} dB (expected: 100 or inf)")
    
    if abs(our_ssim - 1.0) > 1e-6:
        print("   ❌ SSIM self-comparison is not 1.0!")
    else:
        print("   ✅ SSIM calculation OK")
    
    # Compare with skimage (full image, for reference)
    skimage_ssim = ssim_skimage(target_img, target_img, data_range=255)
    print(f"\n   Reference (skimage, full image):")
    print(f"   SSIM = {skimage_ssim:.6f}")
    
    # 9. Save debug report
    report_path = output_dir / "debug_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[8] Debug report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("Pipeline Debug Check Complete")
    print("=" * 60)
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline Debug Script")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Path to data directory")
    parser.add_argument("--sample-id", type=str, default=None,
                        help="Specific sample ID to test")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for debug files")
    
    args = parser.parse_args()
    
    run_full_check(args.data_dir, args.sample_id, args.output_dir)


if __name__ == "__main__":
    main()
