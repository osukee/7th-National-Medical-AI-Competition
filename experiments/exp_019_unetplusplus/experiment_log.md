# exp_019: U-Net++ Architecture

> 実施日: 2026-01-14

## 仮説

1. **U-Net++** の Dense Skip Connection により、マルチスケール特徴融合が改善される
2. 細かいエッジやテクスチャの保持が向上し、SSIMが改善する

## 変更点

| 設定 | exp_017e (baseline) | exp_019 |
|------|---------------------|---------|
| architecture | unet | **unetplusplus** |
| loss_type | optimized | optimized |
| ssim_weight (β) | 1.0 | 1.0 |
| grad_weight (γ) | 0.5 | 0.5 |
| encoder | efficientnet-b4 | efficientnet-b4 |

## 結果

| Metric | exp_017e | exp_019 | 変化 |
|--------|----------|---------|------|
| **🎯 LB Score** | 0.4289 | - | - |
| CV SSIM | 0.702 | - | - |
| CV PSNR | 15.45 | - | - |

## 考察

(実験完了後に記載)

## 次のアクション

- [ ] 結果を記録
- [ ] 効果があれば EfficientNet-B5 も試す
