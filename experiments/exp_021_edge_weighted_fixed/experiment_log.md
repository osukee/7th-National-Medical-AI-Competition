# exp_021: Fixed EdgeWeightedLoss

> 実施日: 2026-01-15

## 仮説

exp_020 の失敗原因を修正した EdgeWeightedLoss をテスト:
- 旧実装: L1 + エッジ重み付けのみ → worst samples で崩壊
- 新実装: **L1 + SSIM + GradLoss** をすべてエッジ重み付け

## 変更点

| 設定 | exp_019 (best) | exp_021 |
|------|----------------|---------|
| loss_type | optimized | **edge_weighted (FIXED)** |
| lambda_edge | - | 2.0 |
| 内部構成 | L1+SSIM+Grad+TV | **Edge-weighted L1 + SSIM + Grad** |

## 結果

| Metric | exp_019 | exp_021 | 変化 |
|--------|---------|---------|------|
| **🎯 LB Score** | 0.4307 | - | - |
| CV SSIM | 0.704 | - | - |

## 考察

(実験完了後に記載)
