# exp_022: Data Augmentation Enhancement

> 実施日: 2026-01-20

## 仮説

Qiita記事シリーズ参考（記事①）:
1. **Augmentation未実装** → 過学習しやすく汎化性能が低下
2. **albumentationsベースの変換** → 入力/ターゲット同期でImage-to-Image対応
3. 期待効果: LB +0.01〜0.02

## 変更点

| 設定 | exp_019/021 (best) | exp_022 |
|------|--------------------|---------| 
| augmentation_enabled | False | **True** |
| augmentation_strength | - | **0.5** |

### Augmentation構成

```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift=0.05, scale=0.1, rotate=15, p=0.5),
    A.RandomBrightnessContrast(brightness=0.1, contrast=0.1, p=0.5),
    A.GaussNoise(var=(5, 20), p=0.25),
], additional_targets={'target': 'image', 'mask': 'mask'})
```

## 結果

| Metric | exp_019 | exp_022 | 変化 |
|--------|---------|---------|------|
| **🎯 LB Score** | 0.4307 | - | - |
| CV SSIM | 0.704 | - | - |
| CV PSNR | 15.49 | - | - |

## 考察

(実験完了後に記載)

## 次のアクション

- [ ] exp_023: EfficientNet-B5 encoder upgrade
- [ ] exp_024: 3ch入力 (グレースケール+エッジ+コントラスト)
