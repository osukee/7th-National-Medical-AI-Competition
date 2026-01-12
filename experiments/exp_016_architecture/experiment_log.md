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

## 期待効果

- SSIM: +0.02~0.03
- LB Score: +0.02~0.03 (0.43目標)

## 結果

*実行待ち*
