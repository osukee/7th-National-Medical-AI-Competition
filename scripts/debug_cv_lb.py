"""
CV-LB Discrepancy Debug Script

このスクリプトは以下を検証する:
1. CVとLBで同一の計算（uint8 + data_range=255）を行う
2. mask外の値がLB評価に影響するかを検証
3. 1枚の画像でstep-by-stepで確認

使用方法: python debug_cv_lb.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path("/kaggle/input/medical-ai-contest-7th-2025")
OUTPUT_DIR = Path("/kaggle/working")

# ローカルテスト用のパス（必要に応じて変更）
if not DATA_DIR.exists():
    DATA_DIR = Path("c:/Users/osuke_main/7th National Medical AI Competition/data")
    OUTPUT_DIR = Path("c:/Users/osuke_main/7th National Medical AI Competition/debug_output")
    OUTPUT_DIR.mkdir(exist_ok=True)

# ==============================================================================
# Debug Functions
# ==============================================================================

def load_sample(idx=0):
    """訓練データの1サンプルを読み込む"""
    train_csv = DATA_DIR / "train.csv"
    df = pd.read_csv(train_csv)
    
    row = df.iloc[idx]
    sample_id = row["id"]
    
    # 入力画像
    input_path = DATA_DIR / row["input_path"]
    input_img = Image.open(input_path).convert("L")
    input_arr = np.array(input_img)
    
    # GT（ターゲット）
    target_path = DATA_DIR / row["target_path"]
    target_img = Image.open(target_path).convert("L")
    target_arr = np.array(target_img)
    
    # マスク（存在する場合）
    mask_arr = None
    if "mask_path" in row and pd.notna(row.get("mask_path", None)):
        mask_path = DATA_DIR / row["mask_path"]
        if mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask_img) > 0
    
    return {
        "id": sample_id,
        "input": input_arr,
        "target": target_arr,
        "mask": mask_arr,
        "category": row.get("category", "unknown"),
    }


def simulate_submission_pipeline(pred_float):
    """提出パイプラインをシミュレート: float[0,1] → uint8[0,255]"""
    # Step 1: clamp to [0, 1]
    pred_clipped = np.clip(pred_float, 0, 1)
    
    # Step 2: scale to [0, 255] and convert to uint8
    pred_uint8 = (pred_clipped * 255).astype(np.uint8)
    
    return pred_uint8


def calculate_ssim_psnr_variants(pred, target, mask=None):
    """
    複数のバリエーションでSSIM/PSNRを計算
    
    返り値: dict of metrics
    """
    results = {}
    
    # === Variant 1: float [0,1], data_range=1.0 (現在のCV) ===
    pred_f = pred.astype(np.float32) / 255.0
    target_f = target.astype(np.float32) / 255.0
    
    results["cv_float_dr1"] = {
        "ssim": ssim(target_f, pred_f, data_range=1.0),
        "psnr": psnr(target_f, pred_f, data_range=1.0),
    }
    
    # === Variant 2: uint8 [0,255], data_range=255 (推定LB) ===
    results["lb_uint8_dr255"] = {
        "ssim": ssim(target, pred, data_range=255),
        "psnr": psnr(target, pred, data_range=255),
    }
    
    # === Variant 3: mask適用版 (uint8, data_range=255) ===
    if mask is not None:
        # マスク内のみで評価
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        # SSIMはwindowベースなのでmask適用が難しい
        # PSNR（MSEベース）はマスク内のみで計算可能
        mse = np.mean((pred_masked.astype(float) - target_masked.astype(float)) ** 2)
        psnr_masked = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        
        results["lb_masked_psnr"] = {
            "psnr": psnr_masked,
        }
    
    return results


def debug_single_sample(idx=0):
    """1サンプルで詳細デバッグ"""
    print(f"\n{'='*60}")
    print(f"DEBUG: Sample {idx}")
    print(f"{'='*60}")
    
    # サンプル読み込み
    sample = load_sample(idx)
    print(f"ID: {sample['id']}")
    print(f"Category: {sample['category']}")
    print(f"Input shape: {sample['input'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Mask: {'Available' if sample['mask'] is not None else 'None'}")
    
    # ダミー予測（GTをそのまま使用 = 完璧な予測の場合）
    perfect_pred = sample["target"].copy()
    
    # === Test 1: 完璧な予測 ===
    print(f"\n--- Test 1: Perfect Prediction (pred = target) ---")
    metrics = calculate_ssim_psnr_variants(perfect_pred, sample["target"], sample["mask"])
    for variant, values in metrics.items():
        print(f"  {variant}: {values}")
    
    # === Test 2: submissionパイプライン経由 ===
    print(f"\n--- Test 2: Through Submission Pipeline ---")
    # float変換 → submission pipeline → 評価
    pred_float = sample["target"].astype(np.float32) / 255.0
    pred_uint8 = simulate_submission_pipeline(pred_float)
    
    print(f"  pred_float: min={pred_float.min():.4f}, max={pred_float.max():.4f}")
    print(f"  pred_uint8: min={pred_uint8.min()}, max={pred_uint8.max()}")
    
    metrics = calculate_ssim_psnr_variants(pred_uint8, sample["target"], sample["mask"])
    for variant, values in metrics.items():
        print(f"  {variant}: {values}")
    
    # === Test 3: mask外を0埋めした場合 ===
    if sample["mask"] is not None:
        print(f"\n--- Test 3: Mask Outside = 0 ---")
        pred_zero_outside = sample["target"].copy()
        pred_zero_outside[~sample["mask"]] = 0
        
        metrics = calculate_ssim_psnr_variants(pred_zero_outside, sample["target"], None)
        for variant, values in metrics.items():
            print(f"  {variant}: {values}")
    
    # === Value Range Check ===
    print(f"\n--- Value Ranges ---")
    print(f"  Input:  min={sample['input'].min()}, max={sample['input'].max()}, mean={sample['input'].mean():.1f}")
    print(f"  Target: min={sample['target'].min()}, max={sample['target'].max()}, mean={sample['target'].mean():.1f}")
    if sample["mask"] is not None:
        mask_ratio = sample["mask"].sum() / sample["mask"].size
        print(f"  Mask:   coverage={mask_ratio*100:.1f}%")


def main():
    """メインデバッグルーチン"""
    print("=" * 60)
    print("CV-LB Discrepancy Debug Script")
    print("=" * 60)
    
    # 複数サンプルでテスト
    for idx in [0, 10, 50]:
        try:
            debug_single_sample(idx)
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
    
    print("\n" + "=" * 60)
    print("Debug Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
