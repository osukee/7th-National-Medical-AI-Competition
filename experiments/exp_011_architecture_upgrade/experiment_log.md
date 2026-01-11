# exp_011: Architecture Upgrade

> 作成日: 2026-01-11

## 変更内容

| 項目 | 変更前 (exp_010) | 変更後 (exp_011) |
|------|-----------------|-----------------|
| Encoder | resnet34 | **efficientnet-b4** |
| Decoder Attention | なし | **scSE** |
| Resolution | 512 | **384** |
| Loss | EdgeAwareLoss (0.1) | EdgeAwareLoss (0.1) |
| Epochs | 15 | 15 |

## 変更理由

User フィードバックに基づくアーキテクチャ強化:

1. **Encoder強化**: resnet34の表現能力限界 → efficientnet-b4でより高い周波数成分を捉える
2. **Decoder強化**: 標準decoderはupsample+convのみ → scSE attentionでrecalibration
3. **解像度変更**: 512→384で学習高速化、TTA/ensemble余地確保

## 期待値

| 指標 | exp_010 | 期待 (exp_011) | 改善幅 |
|------|---------|---------------|-------|
| SSIM | 0.689 | 0.716+ | +0.03 |
| PSNR | 15.2 dB | 16.2+ dB | +1.0 |
| PB Score | 0.411 | **0.43+** | +0.02 |

## 結果

> ⏳ Kaggle実行待ち...

| 指標 | 結果 | 評価 |
|------|------|------|
| CV SSIM | - | - |
| CV PSNR | - | - |
| PB Score | - | - |

## メモ

- もし期待値に達しない場合 → アーキテクチャ自体を捨てる判断
- 次の手: 768解像度 or U-Net++
