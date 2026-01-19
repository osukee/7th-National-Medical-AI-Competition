"""
Outlier Analysis Script for Medical AI Competition

Purpose: Identify worst-performing samples and classify failure patterns
to guide targeted improvements.

Usage:
    python scripts/analyze_outliers.py --predictions-dir outputs/val_predictions

Analysis:
1. Per-sample SSIM/PSNR computation
2. Distribution statistics and histograms
3. Worst N samples visualization
4. Failure pattern classification
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("medical-ai-contest-7th-2025")
OUTPUT_DIR = Path("outputs/outlier_analysis")


# =============================================================================
# Metric Calculation (LB-Aligned)
# =============================================================================

def calculate_ssim_masked(pred, target, mask, data_range=255):
    """Calculate SSIM only on masked region (LB-aligned)."""
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
    """Calculate PSNR only on masked region (LB-aligned)."""
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
    """Calculate LB score using official Kaggle formula."""
    psnr_norm = np.clip((psnr_val - 15) / 20, 0, 1)
    return (ssim_val + psnr_norm) / 2


# =============================================================================
# Failure Pattern Classification
# =============================================================================

def classify_failure(pred, target, mask):
    """
    Classify the type of failure for a prediction.
    
    Categories:
    - too_dark: Prediction mean is significantly lower than target
    - too_bright: Prediction mean is significantly higher than target
    - blurry: Low edge content compared to target
    - noisy: High frequency noise detected
    - low_contrast: Prediction has much lower std than target
    - good: Prediction is within acceptable range
    """
    mask_bool = mask > 0
    
    if mask_bool.sum() == 0:
        return "no_mask"
    
    pred_m = pred[mask_bool].astype(np.float64)
    target_m = target[mask_bool].astype(np.float64)
    
    # Mean difference
    mean_diff = pred_m.mean() - target_m.mean()
    
    # Std difference
    std_ratio = pred_m.std() / (target_m.std() + 1e-6)
    
    # Edge content (using Laplacian)
    pred_lap = cv2.Laplacian(pred, cv2.CV_64F)
    target_lap = cv2.Laplacian(target, cv2.CV_64F)
    
    pred_edge = np.abs(pred_lap[mask_bool]).mean()
    target_edge = np.abs(target_lap[mask_bool]).mean()
    edge_ratio = pred_edge / (target_edge + 1e-6)
    
    # Classify
    issues = []
    
    if mean_diff < -20:
        issues.append("too_dark")
    elif mean_diff > 20:
        issues.append("too_bright")
    
    if std_ratio < 0.7:
        issues.append("low_contrast")
    
    if edge_ratio < 0.6:
        issues.append("blurry")
    elif edge_ratio > 1.5:
        issues.append("noisy")
    
    if not issues:
        issues.append("other")
    
    return {
        "categories": issues,
        "mean_diff": float(mean_diff),
        "std_ratio": float(std_ratio),
        "edge_ratio": float(edge_ratio),
    }


# =============================================================================
# Visualization
# =============================================================================

def create_comparison_image(pred, target, mask, sample_id, ssim_val, psnr_val, output_path):
    """Create side-by-side comparison image with metrics overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Input (placeholder - using target as we don't have input here)
    axes[0].imshow(target, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Target (Ground Truth)")
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(pred, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Prediction")
    axes[1].axis('off')
    
    # Difference (abs)
    diff = np.abs(pred.astype(np.float32) - target.astype(np.float32))
    axes[2].imshow(diff, cmap='hot', vmin=0, vmax=100)
    axes[2].set_title(f"Abs Difference (max={diff.max():.1f})")
    axes[2].axis('off')
    
    # Masked difference
    masked_diff = diff * (mask > 0)
    axes[3].imshow(masked_diff, cmap='hot', vmin=0, vmax=100)
    axes[3].set_title("Masked Difference")
    axes[3].axis('off')
    
    # Add metrics
    lb_score = calculate_lb_score(ssim_val, psnr_val)
    fig.suptitle(f"{sample_id} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f} dB | LB: {lb_score:.4f}", 
                 fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_distribution_plot(results_df, output_path):
    """Create histogram of SSIM and PSNR distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # SSIM histogram
    axes[0, 0].hist(results_df['ssim'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(results_df['ssim'].mean(), color='red', linestyle='--', 
                       label=f"Mean: {results_df['ssim'].mean():.4f}")
    axes[0, 0].axvline(results_df['ssim'].median(), color='green', linestyle='--',
                       label=f"Median: {results_df['ssim'].median():.4f}")
    axes[0, 0].set_xlabel("SSIM")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("SSIM Distribution")
    axes[0, 0].legend()
    
    # PSNR histogram
    axes[0, 1].hist(results_df['psnr'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(results_df['psnr'].mean(), color='red', linestyle='--',
                       label=f"Mean: {results_df['psnr'].mean():.2f}")
    axes[0, 1].axvline(results_df['psnr'].median(), color='green', linestyle='--',
                       label=f"Median: {results_df['psnr'].median():.2f}")
    axes[0, 1].set_xlabel("PSNR (dB)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("PSNR Distribution")
    axes[0, 1].legend()
    
    # LB Score histogram
    axes[1, 0].hist(results_df['lb_score'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(results_df['lb_score'].mean(), color='red', linestyle='--',
                       label=f"Mean: {results_df['lb_score'].mean():.4f}")
    axes[1, 0].set_xlabel("LB Score")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("LB Score Distribution")
    axes[1, 0].legend()
    
    # Category breakdown (if available)
    if 'category' in results_df.columns:
        category_means = results_df.groupby('category')['ssim'].mean().sort_values()
        axes[1, 1].barh(category_means.index, category_means.values)
        axes[1, 1].set_xlabel("Mean SSIM")
        axes[1, 1].set_title("SSIM by Category")
    else:
        axes[1, 1].text(0.5, 0.5, "No category info", ha='center', va='center')
        axes[1, 1].set_title("SSIM by Category (N/A)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_validation_predictions(data_dir, predictions_dir, output_dir, worst_n=20):
    """
    Analyze validation predictions and identify outliers.
    
    Args:
        data_dir: Path to data directory with train.csv and images
        predictions_dir: Path to directory with prediction images (sample_id.png)
        output_dir: Output directory for analysis results
        worst_n: Number of worst samples to visualize
    """
    data_dir = Path(data_dir)
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Outlier Analysis")
    print("=" * 60)
    
    # Load train.csv
    train_df = pd.read_csv(data_dir / "train.csv")
    print(f"Loaded {len(train_df)} training samples")
    
    # Find prediction files
    pred_files = list(predictions_dir.glob("*.png"))
    if not pred_files:
        pred_files = list(predictions_dir.glob("*.npy"))
    
    if not pred_files:
        print(f"No prediction files found in {predictions_dir}")
        print("Expected format: sample_id.png or sample_id.npy")
        return None
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Compute per-sample metrics
    results = []
    failure_patterns = defaultdict(int)
    
    for pred_file in pred_files:
        sample_id = pred_file.stem
        
        # Find corresponding sample in train_df
        sample_row = train_df[train_df['id'] == sample_id]
        if len(sample_row) == 0:
            continue
        sample_row = sample_row.iloc[0]
        
        # Load prediction
        if pred_file.suffix == '.npy':
            pred = np.load(pred_file)
        else:
            pred = np.array(Image.open(pred_file).convert('L'))
        
        # Load target
        target_path = data_dir / sample_row['target_path']
        target = np.array(Image.open(target_path).convert('L'))
        
        # Load mask
        mask_path = data_dir / sample_row['mask_path']
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Ensure same size
        if pred.shape != target.shape:
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]))
        
        # Calculate metrics
        ssim_val = calculate_ssim_masked(pred, target, mask)
        psnr_val = calculate_psnr_masked(pred, target, mask)
        lb_score = calculate_lb_score(ssim_val, psnr_val)
        
        # Classify failure
        failure = classify_failure(pred, target, mask)
        for cat in failure['categories']:
            failure_patterns[cat] += 1
        
        results.append({
            'sample_id': sample_id,
            'category': sample_row.get('category', 'unknown'),
            'ssim': ssim_val,
            'psnr': psnr_val,
            'lb_score': lb_score,
            'failure_categories': failure['categories'],
            'mean_diff': failure['mean_diff'],
            'std_ratio': failure['std_ratio'],
            'edge_ratio': failure['edge_ratio'],
        })
    
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Samples analyzed: {len(results_df)}")
    print(f"\nSSIM:")
    print(f"  Mean:   {results_df['ssim'].mean():.4f}")
    print(f"  Std:    {results_df['ssim'].std():.4f}")
    print(f"  Min:    {results_df['ssim'].min():.4f}")
    print(f"  Max:    {results_df['ssim'].max():.4f}")
    print(f"  5%:     {results_df['ssim'].quantile(0.05):.4f}")
    print(f"  95%:    {results_df['ssim'].quantile(0.95):.4f}")
    
    print(f"\nPSNR:")
    print(f"  Mean:   {results_df['psnr'].mean():.2f} dB")
    print(f"  Std:    {results_df['psnr'].std():.2f} dB")
    print(f"  Min:    {results_df['psnr'].min():.2f} dB")
    print(f"  Max:    {results_df['psnr'].max():.2f} dB")
    
    print(f"\nLB Score:")
    print(f"  Mean:   {results_df['lb_score'].mean():.4f}")
    print(f"  Std:    {results_df['lb_score'].std():.4f}")
    
    print(f"\nFailure Pattern Distribution:")
    for pattern, count in sorted(failure_patterns.items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count} ({count/len(results_df)*100:.1f}%)")
    
    # Category breakdown
    if 'category' in results_df.columns:
        print(f"\nCategory Breakdown:")
        for cat in sorted(results_df['category'].unique()):
            cat_df = results_df[results_df['category'] == cat]
            print(f"  {cat}: n={len(cat_df)}, SSIM={cat_df['ssim'].mean():.4f}, PSNR={cat_df['psnr'].mean():.2f}")
    
    # Save results
    results_df.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    print(f"\nResults saved to: {output_dir / 'per_sample_metrics.csv'}")
    
    # Create distribution plot
    create_distribution_plot(results_df, output_dir / "distribution.png")
    print(f"Distribution plot saved to: {output_dir / 'distribution.png'}")
    
    # Identify and visualize worst samples
    print(f"\n{'='*60}")
    print(f"Worst {worst_n} Samples")
    print(f"{'='*60}")
    
    worst_samples = results_df.nsmallest(worst_n, 'lb_score')
    
    worst_dir = output_dir / "worst_samples"
    worst_dir.mkdir(exist_ok=True)
    
    for i, row in worst_samples.iterrows():
        sample_id = row['sample_id']
        print(f"{row['sample_id']}: SSIM={row['ssim']:.4f}, PSNR={row['psnr']:.2f}, "
              f"LB={row['lb_score']:.4f}, failures={row['failure_categories']}")
        
        # Load and create comparison
        pred_file = predictions_dir / f"{sample_id}.png"
        if not pred_file.exists():
            pred_file = predictions_dir / f"{sample_id}.npy"
        
        if pred_file.exists():
            sample_row = train_df[train_df['id'] == sample_id].iloc[0]
            
            if pred_file.suffix == '.npy':
                pred = np.load(pred_file)
            else:
                pred = np.array(Image.open(pred_file).convert('L'))
            
            target = np.array(Image.open(data_dir / sample_row['target_path']).convert('L'))
            mask = np.array(Image.open(data_dir / sample_row['mask_path']).convert('L'))
            
            if pred.shape != target.shape:
                pred = cv2.resize(pred, (target.shape[1], target.shape[0]))
            
            output_path = worst_dir / f"worst_{i:02d}_{sample_id}.png"
            create_comparison_image(pred, target, mask, sample_id, 
                                   row['ssim'], row['psnr'], output_path)
    
    print(f"\nWorst sample comparisons saved to: {worst_dir}")
    
    # Save summary report
    report = {
        'n_samples': len(results_df),
        'ssim_mean': float(results_df['ssim'].mean()),
        'ssim_std': float(results_df['ssim'].std()),
        'psnr_mean': float(results_df['psnr'].mean()),
        'psnr_std': float(results_df['psnr'].std()),
        'lb_score_mean': float(results_df['lb_score'].mean()),
        'failure_patterns': dict(failure_patterns),
        'worst_samples': worst_samples[['sample_id', 'ssim', 'psnr', 'lb_score', 'failure_categories']].to_dict('records'),
    }
    
    with open(output_dir / "analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return results_df


def analyze_from_model_output(data_dir, model_output_dir, output_dir, worst_n=20):
    """
    Alternative: Analyze directly from saved validation outputs.
    
    This is useful when predictions are stored as .npy files during training.
    """
    # This is a wrapper that handles different output formats
    return analyze_validation_predictions(data_dir, model_output_dir, output_dir, worst_n)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Outlier Analysis Script")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Path to data directory")
    parser.add_argument("--predictions-dir", type=str, required=True,
                        help="Path to directory with prediction images")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for analysis results")
    parser.add_argument("--worst-n", type=int, default=20,
                        help="Number of worst samples to visualize")
    
    args = parser.parse_args()
    
    analyze_validation_predictions(
        args.data_dir,
        args.predictions_dir,
        args.output_dir,
        args.worst_n
    )


if __name__ == "__main__":
    main()
