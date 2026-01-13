# exp_018: Test Time Augmentation (TTA)

> 実施日: 2026-01-13

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

## 期待効果

- LB Score: +0.01~0.02 (0.43+ 目標)

## 結果

*実行待ち*
