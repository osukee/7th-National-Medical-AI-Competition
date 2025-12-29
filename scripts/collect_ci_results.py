#!/usr/bin/env python3
"""
collect_ci_results.py - CI結果を収集し、判断材料を生成するスクリプト

Usage:
    python scripts/collect_ci_results.py <experiment_id> [--metrics-file <path>]

Example:
    python scripts/collect_ci_results.py exp_002_augment
    python scripts/collect_ci_results.py exp_002_augment --metrics-file ./metrics.json

Output:
    experiments/<experiment_id>/result.md に転記可能な形式で出力
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def load_metrics(metrics_path: Path) -> dict:
    """metrics.jsonを読み込む"""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_previous_metrics(experiments_dir: Path, current_exp_id: str) -> Optional[dict]:
    """前回の実験結果を読み込む（比較用）"""
    exp_dirs = sorted(experiments_dir.glob("exp_*"))
    
    current_idx = None
    for i, exp_dir in enumerate(exp_dirs):
        if exp_dir.name == current_exp_id:
            current_idx = i
            break
    
    if current_idx is None or current_idx == 0:
        return None
    
    # 前回の実験を探す
    prev_exp_dir = exp_dirs[current_idx - 1]
    prev_metrics_path = prev_exp_dir / "metrics.json"
    
    if prev_metrics_path.exists():
        with open(prev_metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    return None


def interpret_ssim(ssim: float) -> str:
    """SSIM値を解釈"""
    if ssim >= 0.95:
        return "優秀 - 現状維持 or 微調整"
    elif ssim >= 0.90:
        return "良好 - 改善余地あり"
    elif ssim >= 0.80:
        return "要改善 - アーキテクチャ見直し推奨"
    else:
        return "不良 - 根本的な問題調査が必要"


def interpret_psnr(psnr: float) -> str:
    """PSNR値を解釈"""
    if psnr >= 40:
        return "優秀 - 現状維持 or 微調整"
    elif psnr >= 35:
        return "良好 - 改善余地あり"
    elif psnr >= 30:
        return "要改善 - 損失関数・アーキテクチャ見直し推奨"
    else:
        return "不良 - 根本的な問題調査が必要"


def calculate_improvement(current: dict, previous: Optional[dict]) -> dict:
    """改善度を計算"""
    if previous is None:
        return {"ssim_diff": None, "psnr_diff": None, "is_baseline": True}
    
    current_metrics = current.get("metrics", {})
    prev_metrics = previous.get("metrics", {})
    
    ssim_diff = current_metrics.get("ssim", 0) - prev_metrics.get("ssim", 0)
    psnr_diff = current_metrics.get("psnr", 0) - prev_metrics.get("psnr", 0)
    
    return {
        "ssim_diff": ssim_diff,
        "psnr_diff": psnr_diff,
        "is_baseline": False
    }


def determine_decision(current: dict, improvement: dict) -> tuple[str, str]:
    """採用/却下/保留を判断"""
    ci_status = current.get("ci_status", "unknown")
    
    if ci_status != "passed":
        return "却下", f"CI失敗 ({ci_status})"
    
    if improvement["is_baseline"]:
        return "採用（ベースライン）", "基準値として採用"
    
    ssim_diff = improvement["ssim_diff"]
    psnr_diff = improvement["psnr_diff"]
    
    # 採用条件: SSIM +0.005以上 AND PSNR +0.5dB以上
    if ssim_diff >= 0.005 and psnr_diff >= 0.5:
        return "採用", f"SSIM +{ssim_diff:.4f}, PSNR +{psnr_diff:.2f} dB の改善"
    
    # 却下条件: SSIM -0.01以下 OR PSNR -1dB以下
    if ssim_diff <= -0.01 or psnr_diff <= -1.0:
        return "却下", f"SSIM {ssim_diff:+.4f}, PSNR {psnr_diff:+.2f} dB - 性能低下"
    
    return "保留", f"SSIM {ssim_diff:+.4f}, PSNR {psnr_diff:+.2f} dB - 追加検証推奨"


def generate_report(experiment_id: str, metrics: dict, improvement: dict) -> str:
    """result.md用のレポートを生成"""
    m = metrics.get("metrics", {})
    ssim = m.get("ssim", 0)
    psnr = m.get("psnr", 0)
    ssim_std = m.get("ssim_std", 0)
    psnr_std = m.get("psnr_std", 0)
    ci_status = metrics.get("ci_status", "unknown")
    training_time = metrics.get("training_time_seconds", 0)
    gpu_memory = metrics.get("gpu_memory_peak_mb", 0)
    
    decision, reason = determine_decision(metrics, improvement)
    
    report = f"""# 結果

## 実験ID

{experiment_id}

## CI結果

| 指標 | 値 | 標準偏差 |
|------|-----|----------|
| SSIM | {ssim:.4f} | ±{ssim_std:.4f} |
| PSNR | {psnr:.2f} dB | ±{psnr_std:.2f} |
| Training Time | {training_time} sec | - |
| GPU Memory Peak | {gpu_memory} MB | - |

CI Status: {"✅ Pass" if ci_status == "passed" else "❌ Fail"}

## 指標の解釈

- **SSIM ({ssim:.4f})**: {interpret_ssim(ssim)}
- **PSNR ({psnr:.2f} dB)**: {interpret_psnr(psnr)}

"""
    
    if not improvement["is_baseline"]:
        report += f"""## 前回実験との比較

| 指標 | 差分 | 判定 |
|------|------|------|
| SSIM | {improvement["ssim_diff"]:+.4f} | {"✅ 改善" if improvement["ssim_diff"] > 0 else "❌ 悪化" if improvement["ssim_diff"] < 0 else "➖ 変化なし"} |
| PSNR | {improvement["psnr_diff"]:+.2f} dB | {"✅ 改善" if improvement["psnr_diff"] > 0 else "❌ 悪化" if improvement["psnr_diff"] < 0 else "➖ 変化なし"} |

"""
    
    report += f"""## 判断

**{decision}**

理由: {reason}

## 分析

### 良かった点

- (分析結果を記載)

### 悪かった点

- (分析結果を記載)

## 次のアクション

1. (次に試すべき仮説を記載)

---

*Generated at {datetime.now().isoformat()}*
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="CI結果を収集し、判断材料を生成する"
    )
    parser.add_argument(
        "experiment_id",
        help="実験ID (例: exp_002_augment)"
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="metrics.jsonのパス (デフォルト: ./metrics.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力先 (デフォルト: 標準出力)"
    )
    
    args = parser.parse_args()
    
    # metrics.jsonのパスを決定
    if args.metrics_file:
        metrics_path = args.metrics_file
    else:
        metrics_path = Path("./metrics.json")
    
    try:
        # メトリクスを読み込み
        metrics = load_metrics(metrics_path)
        
        # 前回の実験結果を読み込み（比較用）
        experiments_dir = Path("./experiments")
        previous_metrics = load_previous_metrics(experiments_dir, args.experiment_id)
        
        # 改善度を計算
        improvement = calculate_improvement(metrics, previous_metrics)
        
        # レポートを生成
        report = generate_report(args.experiment_id, metrics, improvement)
        
        # 出力
        if args.output:
            args.output.write_text(report, encoding="utf-8")
            print(f"Report saved to: {args.output}")
        else:
            print(report)
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in metrics file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
