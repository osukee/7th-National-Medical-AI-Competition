# exp_020: Edge-Weighted Loss

> 実施日: 2026-01-14

## 仮説

1. **Edge-Weighted Loss** でエッジ領域の誤差を強調することでSSIMが改善する
2. 上位論文で採用されている手法: `pixel_weight = 1 + λ * normalized_edge(input)`

## 変更点

| 設定 | exp_019 (best) | exp_020 |
|------|----------------|---------|
| architecture | unetplusplus | unetplusplus |
| loss_type | optimized | **edge_weighted** |
| lambda_edge | - | **2.0** |

## 結果 ❌

| Metric | exp_019 | exp_020 | 変化 |
|--------|---------|---------|------|
| **🎯 LB Score** | 0.4307 | **0.4120** | **-0.0187** ❌ |
| CV SSIM | 0.704 | 0.702 | -0.002 |
| worst_eval SSIM | 0.92 | **0.37** | **-0.55** 😱 |

## 考察

**大失敗**: EdgeWeightedLoss の実装に問題あり

問題点:
- 現在の実装は L1 + エッジ重み付けのみ
- **SSIM 損失が含まれていない** → worst samples で崩壊
- 上位論文の実装は SSIM も含んでいる

## 結論

exp_019 (OptimizedLoss + U-Net++) に戻す。Edge-weighted は実装修正が必要。
