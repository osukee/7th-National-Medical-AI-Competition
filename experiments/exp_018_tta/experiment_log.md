# exp_018: Test Time Augmentation (TTA)

> 実施日: 2026-01-14

## 仮説

- TTAによる4パターン平均化で予測の安定性向上
- ノイズ軽減・エッジ保持改善

## 変更点 (exp_016ベース)

| 設定 | exp_016 | exp_018 |
|------|---------|---------|
| tta_enabled | - | **True** |

### TTA構成

1. Original
2. Horizontal flip → 予測 → flip back
3. Vertical flip → 予測 → flip back  
4. Both flips → 予測 → flip back
5. 4つの平均を最終予測

## 結果 🎉 成功！

| Metric | exp_016 | exp_018 | 変化 |
|--------|---------|---------|------|
| **🎯 LB Score** | 0.419 | **0.428** | **+0.009** ✅ |
| SSIM (CV) | 0.703 | 0.704 | ≈0 |
| PSNR (CV) | 15.47 | 15.49 | ≈0 |

## 考察

- **TTAは明確に有効** (+0.009 LB)
- CVはほぼ変化なし → TTAの効果は推論時の安定化
- 4-way augmentationで十分な効果

## 累積進捗

| 実験 | 変更 | LB | 増分 |
|------|------|-----|------|
| exp_015 | Baseline Fix | 0.407 | - |
| exp_016 | efficientnet-b4 | 0.419 | +0.012 |
| exp_017 | Loss weight | 0.413 | ❌ |
| **exp_018** | **TTA** | **0.428** | **+0.009** |

**目標0.46まで残り: +0.032**
