# 仮説

## 変更内容

VGG19の特徴量を使用したPerceptual Lossを追加し、構造的な詳細の再現性を向上させる。

**損失関数の変更**:
```python
Loss = L1 + λ1 × SSIM_Loss + λ2 × Perceptual_Loss
```

- λ1 = 1.0 (SSIM Loss重み)
- λ2 = 0.1 (Perceptual Loss重み、初期値)

## 期待する効果

- **SSIM**: ベースラインから +0.02〜0.04 改善
- **PSNR**: ベースラインから +0.5〜1.5 dB 改善
- エッジやテクスチャの鮮明度向上

## 根拠

1. **高レベル特徴の活用**: VGGの中間層特徴量は画像の構造的パターンを捉える
2. **スタイル転送での実績**: 画像生成タスクで広く使用される手法
3. **SSIM直接最適化の補完**: L1/SSIMでは捉えにくいテクスチャ品質を改善

## リスク

- メモリ使用量の増加（VGG19の追加読み込み）
- 学習時間の増加（約1.3倍）
- λ2の調整が困難な可能性
- グレースケール画像へのVGG適用の限界

## 実装

```python
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        # グレースケールを3チャンネルに変換
        pred_3ch = pred.repeat(1, 3, 1, 1)
        target_3ch = target.repeat(1, 3, 1, 1)
        return F.mse_loss(self.vgg(pred_3ch), self.vgg(target_3ch))
```
