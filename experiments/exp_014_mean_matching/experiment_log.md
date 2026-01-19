# exp_014: Mean Matching Only

> 作成日: 2026-01-12

## 変更内容

exp_013 Distribution Analysis から判明した問題:
- **Mean偏移 = −16.1** (予測がtargetより暗い)
- Std比 = 0.899 (許容範囲)

### 対策: Global Mean Matching

```python
# test予測時のみ適用
pred_corrected = pred + 16.1
pred_corrected = clip(pred_corrected, 0, 255)
```

**やらないこと:**
- ❌ std正規化
- ❌ per-sample補正
- ❌ adaptive補正

## 期待値

| 指標 | exp_013 | 期待 (exp_014) |
|------|---------|----------------|
| CV SSIM | 0.687 | ±0.005 (変化なし) |
| CV PSNR | 15.2 dB | +0.3 dB |
| PB Score | 0.411 | **+0.01** |

## 結果

> ⏳ Kaggle実行待ち...

| 指標 | 結果 | 評価 |
|------|------|------|
| CV SSIM | - | - |
| CV PSNR | - | - |
| PB Score | - | - |
