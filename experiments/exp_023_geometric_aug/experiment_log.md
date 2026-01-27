# exp_023: Geometric-Only Augmentation

> 実施日: 2026-01-21

## 仮説

exp_022失敗の根本原因分析:
1. **Brightness/Contrast変更** → 入力-ターゲット対応を破壊
2. **GaussNoise** → 信号を劣化させモデルが学習できない
3. **根本的な問題**: Image-to-Image生成には**幾何変換のみ**が安全

## 変更点

| 設定 | exp_022 (失敗) | exp_023 |
|------|----------------|---------|
| augmentation_mode | 'intensity' | **'geometric'** |

### 新Augmentation構成

```python
A.Compose([
    # 安全な変換のみ
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),           # NEW: 90度回転
    A.ShiftScaleRotate(                # パラメータ縮小
        shift=0.03, scale=0.05, rotate=10, p=0.25
    ),
    A.ElasticTransform(                # NEW: 細胞変形シミュレーション
        alpha=50, sigma=5, p=0.15
    ),
])
# 削除: RandomBrightnessContrast, GaussNoise
```

## 期待効果

- LB +0.01〜0.02 (exp_019 0.4307 → 0.44〜0.45)
- Image-to-Image対応関係を維持しながら汎化性能向上

## 結果 🎉 成功！

| Metric | exp_019 | exp_023 | 変化 |
|--------|---------|---------|------|
| **🎯 LB Score** | 0.4307 | **0.44039** | **+0.010** ✅ |

## 考察

- **幾何変換のみのAugmentationは有効** (+0.010 LB)
- exp_022の輝度/コントラスト変更が失敗した原因をGeometric-onlyで回避
- 汎化性能が向上し、過学習が軽減された

## 累積進捗

| 実験 | 変更 | LB | 増分 |
|------|------|-----|------|
| exp_015 | Baseline Fix | 0.407 | - |
| exp_016 | efficientnet-b4 | 0.419 | +0.012 |
| exp_018 | TTA | 0.428 | +0.009 |
| exp_019 | U-Net++ | 0.4307 | +0.003 |
| exp_022 | Intensity Aug | 0.43035 | ❌ |
| **exp_023** | **Geometric Aug** | **0.44039** | **+0.010** ✅ |

**目標0.46まで残り: +0.020**

## 次のアクション

- [x] Kaggleで実験実行
- [x] 結果をここに記録
- [ ] overview.mdを更新
