# 仮説

## 変更内容

U-NetをAttention U-Net（MAnet）に変更し、重要な領域への注意機構を追加する。

- **architecture**: `Unet` → `MAnet`（Multi-scale Attention Net）

## 期待する効果

- **SSIM**: ベースラインから +0.02〜0.04 改善
- **PSNR**: ベースラインから +1.0〜2.0 dB 改善
- オルガノイド領域への集中により、背景ノイズを削減

## 根拠

1. **Attention機構**: 重要な領域に選択的に注目
2. **マルチスケール特徴**: 異なる解像度の情報を効果的に統合
3. **医療画像での実績**: セグメンテーションタスクで高い精度
4. **SMP対応**: segmentation_models_pytorchで直接利用可能

## リスク

- 学習時間の微増
- メモリ使用量の若干の増加
- Image-to-Imageタスクへの適合性は未検証
