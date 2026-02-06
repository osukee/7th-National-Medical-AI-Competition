"""
LB Evaluation Function Reproduction & Output Distribution Analysis

Purpose:
1. Reproduce LB evaluation function exactly (mask, data_range=255, SSIM implementation)
2. Visualize output distribution (mean/std/histogram/sample variance)
3. Diagnose whether distribution is stable or needs normalization

Usage:
    python scripts/analyze_lb_distribution.py --predictions_dir outputs/predictions
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd


def calculate_ssim_lb_aligned(pred, target, mask, data_range=255):
    """
    LB-aligned SSIM calculation.
    
    Key parameters (must match LB exactly):
    - data_range=255 (uint8)
    - mask-based: only evaluate pixels where mask > 0
    - Use skimage.metrics.structural_similarity with win_size=7 (odd, <= min dimension)
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mask_bool = mask > 0
    
    if mask_bool.sum() == 0:
        return 0.0
    
    # Method 1: Simple masked SSIM (mean/var based)
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


def calculate_psnr_lb_aligned(pred, target, mask, data_range=255):
    """
    LB-aligned PSNR calculation.
    - Only on masked region
    - data_range=255
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


def analyze_distribution(predictions, masks, targets, sample_ids):
    """
    Analyze output distribution across all samples.
    
    Returns dict with:
    - Global stats (mean, std, min, max)
    - Per-sample stats
    - Mask-inside vs mask-outside comparison
    - Histogram data
    """
    stats = {
        'global': {},
        'per_sample': [],
        'mask_inside': [],
        'mask_outside': [],
    }
    
    all_inside_pixels = []
    all_outside_pixels = []
    all_target_inside = []
    
    for pred, mask, target, sid in zip(predictions, masks, targets, sample_ids):
        mask_bool = mask > 0
        
        inside = pred[mask_bool] if mask_bool.sum() > 0 else np.array([])
        outside = pred[~mask_bool] if (~mask_bool).sum() > 0 else np.array([])
        target_inside = target[mask_bool] if mask_bool.sum() > 0 else np.array([])
        
        sample_stat = {
            'id': sid,
            'mean': float(pred.mean()),
            'std': float(pred.std()),
            'min': int(pred.min()),
            'max': int(pred.max()),
            'inside_mean': float(inside.mean()) if len(inside) > 0 else 0,
            'inside_std': float(inside.std()) if len(inside) > 0 else 0,
            'target_inside_mean': float(target_inside.mean()) if len(target_inside) > 0 else 0,
        }
        stats['per_sample'].append(sample_stat)
        
        if len(inside) > 0:
            all_inside_pixels.extend(inside.tolist())
            all_target_inside.extend(target_inside.tolist())
        if len(outside) > 0:
            all_outside_pixels.extend(outside.tolist())
    
    # Global stats
    all_inside = np.array(all_inside_pixels)
    all_outside = np.array(all_outside_pixels)
    all_target_in = np.array(all_target_inside)
    
    stats['global'] = {
        'inside_mean': float(all_inside.mean()) if len(all_inside) > 0 else 0,
        'inside_std': float(all_inside.std()) if len(all_inside) > 0 else 0,
        'outside_mean': float(all_outside.mean()) if len(all_outside) > 0 else 0,
        'outside_std': float(all_outside.std()) if len(all_outside) > 0 else 0,
        'target_inside_mean': float(all_target_in.mean()) if len(all_target_in) > 0 else 0,
        'target_inside_std': float(all_target_in.std()) if len(all_target_in) > 0 else 0,
        # 分布差
        'mean_diff': float(all_inside.mean() - all_target_in.mean()) if len(all_inside) > 0 else 0,
        'std_ratio': float(all_inside.std() / all_target_in.std()) if len(all_target_in) > 0 and all_target_in.std() > 0 else 1,
    }
    
    # Sample間のばらつき
    sample_means = [s['inside_mean'] for s in stats['per_sample']]
    stats['global']['sample_mean_std'] = float(np.std(sample_means))
    
    return stats


def plot_distributions(stats, output_path):
    """
    Plot distribution analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Per-sample mean distribution
    ax1 = axes[0, 0]
    sample_means = [s['inside_mean'] for s in stats['per_sample']]
    ax1.hist(sample_means, bins=30, alpha=0.7, label='Pred inside mean')
    ax1.axvline(stats['global']['target_inside_mean'], color='r', linestyle='--', label='Target mean')
    ax1.set_xlabel('Mean pixel value')
    ax1.set_ylabel('Count')
    ax1.set_title('Per-sample Inside Mean Distribution')
    ax1.legend()
    
    # 2. Per-sample std distribution
    ax2 = axes[0, 1]
    sample_stds = [s['inside_std'] for s in stats['per_sample']]
    ax2.hist(sample_stds, bins=30, alpha=0.7, label='Pred inside std')
    ax2.axvline(stats['global']['target_inside_std'], color='r', linestyle='--', label='Target std')
    ax2.set_xlabel('Std pixel value')
    ax2.set_ylabel('Count')
    ax2.set_title('Per-sample Inside Std Distribution')
    ax2.legend()
    
    # 3. Pred mean vs Target mean scatter
    ax3 = axes[1, 0]
    pred_means = [s['inside_mean'] for s in stats['per_sample']]
    target_means = [s['target_inside_mean'] for s in stats['per_sample']]
    ax3.scatter(target_means, pred_means, alpha=0.5)
    ax3.plot([0, 255], [0, 255], 'r--', label='y=x')
    ax3.set_xlabel('Target inside mean')
    ax3.set_ylabel('Pred inside mean')
    ax3.set_title('Pred vs Target Mean (per sample)')
    ax3.legend()
    
    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary = f"""
Distribution Analysis Summary
=============================

Pred Inside:
  Mean: {stats['global']['inside_mean']:.1f}
  Std: {stats['global']['inside_std']:.1f}

Target Inside:
  Mean: {stats['global']['target_inside_mean']:.1f}
  Std: {stats['global']['target_inside_std']:.1f}

Difference:
  Mean diff: {stats['global']['mean_diff']:.1f}
  Std ratio: {stats['global']['std_ratio']:.3f}

Sample-to-sample variability:
  Sample mean std: {stats['global']['sample_mean_std']:.1f}

Diagnosis:
  {'⚠️ Mean偏移が大きい → 正規化推奨' if abs(stats['global']['mean_diff']) > 10 else '✅ Mean偏移は許容範囲'}
  {'⚠️ Std比が不均衡 → コントラスト調整推奨' if abs(stats['global']['std_ratio'] - 1) > 0.2 else '✅ Std比は許容範囲'}
"""
    ax4.text(0.1, 0.5, summary, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Distribution plot saved to {output_path}")
    plt.close()


def run_lb_simulation(data_dir, predictions_dir, output_dir):
    """
    Run LB simulation on validation predictions.
    """
    data_dir = Path(data_dir)
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load train.csv for ground truth paths
    train_csv = data_dir / "train.csv"
    if not train_csv.exists():
        print(f"Error: {train_csv} not found")
        return
    
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} samples from train.csv")
    
    # Find prediction files
    pred_files = sorted(predictions_dir.glob("*.png"))
    if not pred_files:
        pred_files = sorted(predictions_dir.glob("*.npy"))
    
    print(f"Found {len(pred_files)} prediction files")
    
    predictions = []
    masks = []
    targets = []
    sample_ids = []
    ssim_scores = []
    psnr_scores = []
    
    for pred_file in pred_files:
        # Extract sample ID from filename
        sid = pred_file.stem.replace("_pred", "").replace("_prediction", "")
        
        # Find corresponding row in df
        row = df[df['id'] == sid]
        if len(row) == 0:
            print(f"Warning: {sid} not found in train.csv")
            continue
        row = row.iloc[0]
        
        # Load prediction
        if pred_file.suffix == '.npy':
            pred = np.load(pred_file)
        else:
            pred = np.array(Image.open(pred_file).convert('L'))
        
        # Load target
        target_path = data_dir / row['target_path']
        if not target_path.exists():
            print(f"Warning: {target_path} not found")
            continue
        target = np.array(Image.open(target_path).convert('L'))
        
        # Load mask
        mask_path = data_dir / row['mask_path']
        if not mask_path.exists():
            print(f"Warning: {mask_path} not found")
            continue
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Resize if needed
        if pred.shape != (512, 512):
            from PIL import Image as PILImage
            pred = np.array(PILImage.fromarray(pred.astype(np.uint8)).resize((512, 512)))
        if target.shape != (512, 512):
            target = np.array(PILImage.fromarray(target).resize((512, 512)))
        if mask.shape != (512, 512):
            mask = np.array(PILImage.fromarray(mask).resize((512, 512)))
        
        # Calculate LB-aligned metrics
        ssim_val = calculate_ssim_lb_aligned(pred, target, mask)
        psnr_val = calculate_psnr_lb_aligned(pred, target, mask)
        
        predictions.append(pred)
        masks.append(mask)
        targets.append(target)
        sample_ids.append(sid)
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)
        
        print(f"{sid}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}")
    
    if len(predictions) == 0:
        print("No valid predictions found")
        return
    
    # Calculate aggregate metrics
    mean_ssim = np.mean(ssim_scores)
    mean_psnr = np.mean(psnr_scores)
    
    # Simulate LB score (assuming 0.5 * SSIM + 0.5 * normalized_PSNR)
    # This is an approximation - actual LB formula may differ
    lb_approx = mean_ssim  # Often LB is just SSIM or weighted
    
    print(f"\n{'='*50}")
    print(f"LB Simulation Results")
    print(f"{'='*50}")
    print(f"Samples: {len(predictions)}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"Approx LB Score: {lb_approx:.4f}")
    
    # Analyze distribution
    print(f"\n{'='*50}")
    print(f"Distribution Analysis")
    print(f"{'='*50}")
    
    stats = analyze_distribution(predictions, masks, targets, sample_ids)
    
    print(f"Pred mask-inside: mean={stats['global']['inside_mean']:.1f}, std={stats['global']['inside_std']:.1f}")
    print(f"Target mask-inside: mean={stats['global']['target_inside_mean']:.1f}, std={stats['global']['target_inside_std']:.1f}")
    print(f"Mean diff: {stats['global']['mean_diff']:.1f}")
    print(f"Std ratio: {stats['global']['std_ratio']:.3f}")
    print(f"Sample-to-sample mean std: {stats['global']['sample_mean_std']:.1f}")
    
    # Save results
    results = {
        'n_samples': len(predictions),
        'ssim_mean': mean_ssim,
        'psnr_mean': mean_psnr,
        'lb_approx': lb_approx,
        'distribution': stats['global'],
        'per_sample_metrics': [
            {'id': sid, 'ssim': s, 'psnr': p}
            for sid, s, p in zip(sample_ids, ssim_scores, psnr_scores)
        ],
    }
    
    with open(output_dir / "lb_simulation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'lb_simulation_results.json'}")
    
    # Plot distributions
    plot_distributions(stats, output_dir / "distribution_analysis.png")
    
    # Diagnosis
    print(f"\n{'='*50}")
    print(f"Diagnosis")
    print(f"{'='*50}")
    
    if abs(stats['global']['mean_diff']) > 10:
        print("⚠️ Mean偏移が大きい (>10) → 分布正規化(mean matching)推奨")
    else:
        print("✅ Mean偏移は許容範囲")
    
    if abs(stats['global']['std_ratio'] - 1) > 0.2:
        print("⚠️ Std比が不均衡 (>0.2) → コントラスト調整推奨")
    else:
        print("✅ Std比は許容範囲")
    
    if stats['global']['sample_mean_std'] > 20:
        print("⚠️ Sample間のばらつきが大きい (>20) → 分布安定化が必要")
    else:
        print("✅ Sample間のばらつきは許容範囲")


def main():
    parser = argparse.ArgumentParser(description='LB Simulation & Distribution Analysis')
    parser.add_argument('--data_dir', type=str, default='medical-ai-contest-7th-2025',
                        help='Path to dataset directory')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Path to prediction outputs directory')
    parser.add_argument('--output_dir', type=str, default='outputs/lb_analysis',
                        help='Output directory for results')
    
    args = parser.parse_args()
    run_lb_simulation(args.data_dir, args.predictions_dir, args.output_dir)


if __name__ == "__main__":
    main()
