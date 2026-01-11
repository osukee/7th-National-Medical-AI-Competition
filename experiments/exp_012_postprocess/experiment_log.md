# exp_012: Post-Processing Pipeline

> 作成日: 2026-01-12

## 変更内容

exp_011の教訓: アーキテクチャ変更だけでは勝てない → **後処理を第一級市民として組み込む**

| 項目 | exp_010 | exp_012 |
|------|---------|---------|
| Model | resnet34, 512 | resnet34, 512 (同じ) |
| Post-processing | なし | **有効化** |

### 後処理パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| pp_smooth_sigma | 1.0 | mask内部の Gaussian smoothing |
| pp_sharpen_strength | 0.3 | mask境界の unsharp mask |
| pp_boundary_width | 5 | 境界領域の幅 (px) |

## 戦略

**「中はぼかす・縁は立てる」**

1. mask 内部 → Gaussian blur で滑らかに
2. mask 境界 → unsharp mask でシャープに
3. 外部 → そのまま

## 期待値

| 指標 | exp_010 | 期待 (exp_012) |
|------|---------|---------------|
| PB Score | 0.411 | **0.42+** |

## 結果

> ⏳ Kaggle実行待ち...

| 指標 | 結果 | 評価 |
|------|------|------|
| CV SSIM | - | - |
| CV PSNR | - | - |
| PB Score | - | - |
