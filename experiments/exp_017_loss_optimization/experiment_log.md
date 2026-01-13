# exp_017: Loss Optimization

> 実施日: 2026-01-14

## Phase A: パイプライン検証結果 ✅

| 項目 | 結果 | 詳細 |
|------|------|------|
| Flatten Order | ✅ 正常 | Row-major (C-order) |
| Float→Uint8 | ✅ 正常 | [0,1]→[0,255] |
| SSIM計算 | ✅ 正常 | data_range=255 |
| PSNR計算 | ✅ 正常 | data_range=255 |
| Mask format | ✅ 正常 | バイナリ（0/255） |

**結論**: パイプラインに問題なし。バグによるスコア低下ではない。

---

## 仮説

1. **OptimizedLoss** (L1 + SSIM + GradLoss + TV) はEdgeAwareLossよりSSIMを改善できる
2. **パラメータチューニング**: β (SSIM重み) と γ (勾配重み) の最適な組み合わせを探索

## 変更点

| 設定 | exp_016 | exp_017 |
|------|---------|---------|
| Loss | EdgeAwareLoss | **OptimizedLoss** |
| ssim_weight (β) | 1.0 | **Grid search: [0.5, 1.0, 1.5]** |
| grad_weight (γ) | 0.1 (edge_weight) | **Grid search: [0.2, 0.5, 1.0]** |
| tv_weight (δ) | N/A | **1e-4** |

## 実験計画

### Grid Search Matrix (3×3 = 9 experiments)

| Exp | β (SSIM) | γ (Grad) | 期待効果 |
|-----|----------|----------|----------|
| 017a | 0.5 | 0.2 | Baseline tuning |
| 017b | 1.0 | 0.2 | Moderate SSIM |
| 017c | 1.5 | 0.2 | Strong SSIM |
| 017d | 0.5 | 0.5 | Balanced |
| 017e | 1.0 | 0.5 | **Recommended** |
| 017f | 1.5 | 0.5 | Strong SSIM + edge |
| 017g | 0.5 | 1.0 | Edge focus |
| 017h | 1.0 | 1.0 | Aggressive |
| 017i | 1.5 | 1.0 | Maximum SSIM |

## 設定変更方法

Config内で以下を変更:

```python
# exp_017: Loss Optimization
loss_type = "optimized"
ssim_weight = 1.0    # β: 0.5, 1.0, or 1.5
grad_weight = 0.5    # γ: 0.2, 0.5, or 1.0
tv_weight = 1e-4     # δ: fixed
```

## 結果

| Exp | β | γ | CV SSIM | CV PSNR | LB Score | 変化 |
|-----|---|---|---------|---------|----------|------|
| 017a | 0.5 | 0.2 | - | - | - | - |
| 017b | 1.0 | 0.2 | - | - | - | - |
| 017c | 1.5 | 0.2 | - | - | - | - |
| 017d | 0.5 | 0.5 | - | - | - | - |
| 017e | 1.0 | 0.5 | - | - | - | - |
| 017f | 1.5 | 0.5 | - | - | - | - |
| 017g | 0.5 | 1.0 | - | - | - | - |
| 017h | 1.0 | 1.0 | - | - | - | - |
| 017i | 1.5 | 1.0 | - | - | - | - |

## 考察

(実験完了後に記載)

## 次のアクション

- [ ] 最良の設定を特定
- [ ] exp_018: U-Net++ アーキテクチャ変更
- [ ] exp_019: EfficientNet-B5 エンコーダ変更
