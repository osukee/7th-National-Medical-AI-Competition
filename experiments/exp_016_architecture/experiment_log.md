# exp_016: Architecture Upgrade

> 実施日: 2026-01-13

## 仮説

1. **efficientnet-b4** はresnet34より特徴抽出能力が高く、SSIM/PSNR両方を改善できる
2. **Mean Matching OFF** により、モデル本来の出力を評価できる

## 変更点

| 設定 | exp_015 | exp_016 |
|------|---------|---------|
| encoder | resnet34 | **efficientnet-b4** |
| mean_matching | ON (+16.1) | **OFF** |

## 結果 🎉

| Metric | exp_015 | exp_016 | 変化 |
|--------|---------|---------|------|
| **🎯 LB Score** | 0.407 | **0.419** | **+0.011** ✅ |
| SSIM (CV) | 0.692 | 0.703 | +0.011 |
| PSNR (CV) | 15.23 | 15.47 | +0.24 |

### カテゴリ別SSIM

| Cat | Mean | 前回 | 変化 |
|-----|------|------|------|
| A | 0.682 | 0.673 | +0.009 |
| B | 0.724 | 0.711 | +0.013 |
| C | 0.703 | 0.692 | +0.011 |

## 考察

- **efficientnet-b4は明確に有効** (+0.011 LB)
- **Mean Matching OFFが正解** （または両方の複合効果）
- 目標0.46まで残り **+0.041**

## 次のアクション

- [ ] Phase 3: Loss Optimization (SSIM weight増加, Edge weight調整)
- [ ] U-Net++ への変更検討
