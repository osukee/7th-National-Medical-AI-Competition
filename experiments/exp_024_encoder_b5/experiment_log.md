# exp_024: EfficientNet-B5 Encoder

> 実施日: 2026-01-27

## 仮説

- EfficientNet-B5はB4より大きなモデルで、より高品質な特徴抽出が可能
- 医療画像のような細かいテクスチャ認識に有利
- 期待改善: +0.005〜0.010

## 変更点 (exp_023ベース)

| 設定 | exp_023 | exp_024 |
|------|---------|---------|
| encoder | efficientnet-b4 | **efficientnet-b5** |

### 継承設定

- architecture: unetplusplus
- augmentation_mode: geometric
- TTA: enabled (4-way flip)

## リスク

- B5はB4より大きいのでメモリ使用量増加
- 学習時間増加
- Kaggle GPUメモリ制限 (16GB) に注意

## 結果

| Metric | exp_023 | exp_024 | 変化 |
|--------|---------|---------|------|
| **🎯 LB Score** | 0.44039 | TBD | TBD |

## 次のアクション

- [ ] Kaggleで実験実行
- [ ] 結果をここに記録
