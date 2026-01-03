# 仮説

## 変更内容

U-NetのencoderをResNet34からEfficientNet-B4に変更する。

- **encoder**: `resnet34` → `efficientnet-b4`
- **batch_size**: 4 → 2（メモリ制約対応）

## 期待する効果

- **SSIM**: ベースラインから +0.01〜0.03 改善
- **PSNR**: ベースラインから +0.5〜1.0 dB 改善
- より効率的な特徴抽出による精度向上

## 根拠

1. **Compound Scaling**: EfficientNetは深さ・幅・解像度を統合的にスケーリング
2. **パラメータ効率**: 少ないパラメータで高い精度を実現
3. **ImageNet精度**: ResNet34より高いImageNet精度（転移学習効果の期待）
4. **SMP対応**: segmentation_models_pytorchで直接利用可能

## リスク

- メモリ使用量増加（batch_size削減が必要）
- グレースケール画像への転移学習効果が限定的な可能性
- 学習時間の増加

## 検証条件

- 同一のデータ・エポック数で比較
- SSIM/PSNRの差が0.01以上で有意と判断
