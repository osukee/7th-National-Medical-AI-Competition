# 仮説

## 変更内容

データ拡張を追加してモデルの汎化性能を向上させる。

- **水平/垂直フリップ**: ランダムに適用
- **回転**: ±15度のランダム回転
- **輝度/コントラスト調整**: ±10%の変動
- **Elastic Deformation**: オルガノイドの形状変動を模倣

## 期待する効果

- **SSIM**: ベースラインから +0.02〜0.05 改善
- **PSNR**: ベースラインから +1〜2 dB 改善

## 根拠

1. **過学習防止**: 限られたデータで多様な変換を学習
2. **位置不変性**: フリップ・回転で方向依存を排除
3. **輝度変動耐性**: 撮影条件の違いに対応
4. **医療画像での実績**: MRIやCTスキャンで広く採用

## リスク

- 過度な拡張は非現実的な画像を生成し性能低下の可能性
- 学習時間の増加（約1.2〜1.5倍）
- 輝度変更が目標画像との不整合を起こす可能性

## 実験設定

```python
# Albumentations使用
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.5
    ),
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),
])
```
