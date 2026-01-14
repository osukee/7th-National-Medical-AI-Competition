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
| ssim_weight | 1.0 | 1.0 |
| grad_weight | 0.5 | 0.5 |

## 結果

| Metric | exp_019 | exp_020 | 変化 |
|--------|---------|---------|------|
| **🎯 LB Score** | 0.4307 | - | - |
| CV SSIM | 0.704 | - | - |
| CV PSNR | 15.49 | - | - |

## 考察

(実験完了後に記載)
