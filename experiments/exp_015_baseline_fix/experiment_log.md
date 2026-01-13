# exp_015: Baseline Fix - 実験結果

> 実施日: 2026-01-12

## 仮説

- 4枚の全ゼロtarget画像を除外することでLoss計算が安定する
- LBスコア向上を期待

## 変更点

1. `EXCLUDED_SAMPLE_IDS`で4サンプル除外
2. `calculate_lb_score()`関数追加
3. `overview.md`に公式評価式追加

## 結果

| Metric | 値 | 前回(exp_008) | 変化 |
|--------|-----|---------------|------|
| **SSIM (CV)** | 0.692 | 0.686 | +0.006 ✅ |
| **PSNR (CV)** | 15.23 dB | 15.2 dB | ≈0 |
| **🎯 LB Score** | **0.407** | **0.410** | **-0.003** ⚠️ |

### カテゴリ別SSIM

| Cat | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Mean |
|-----|-------|-------|-------|-------|-------|------|
| A | 0.660 | 0.688 | 0.685 | 0.671 | 0.659 | 0.673 |
| B | 0.693 | 0.719 | 0.724 | 0.719 | 0.702 | 0.711 |
| C | 0.738 | 0.671 | 0.657 | 0.688 | 0.705 | 0.692 |

### Worst-Case評価

| Metric | 値 |
|--------|-----|
| worst_eval_mean | 0.925 |
| worst_eval_min | 0.845 |

## 考察

- **CV SSIM改善してもLB下がった** → Mean Matching (+16.1) が悪影響している可能性
- LB 0.407 = (SSIM + PSNR_norm) / 2 から逆算:
  - SSIMテスト ≈ 0.72, PSNR_norm ≈ 0.09 (= 16.8dB) と推定
- 目標0.46には +0.053 必要

## 次のアクション

- [ ] **Mean Matching OFF でexp_016** ← 優先
- [ ] Phase 2: Architecture Upgrade (efficientnet-b4, U-Net++)
