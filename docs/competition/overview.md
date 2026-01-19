# 第7回 全国医療AIコンテスト：オルガノイド画像の仮想染色

> [!IMPORTANT]
> このファイルには **事実と制約のみ** を記載する。戦略・感想・憶測は禁止。

---

## 1. タスク概要

| 項目 | 仕様 |
|------|------|
| **タスク種別** | Image-to-Image Generation |
| **入力** | 透過画像（512×512, grayscale） |
| **出力** | FAX蛍光画像（512×512, uint8） |

### データフロー

```
透過画像 (512x512, grayscale) → モデル → FAX蛍光画像 (512x512, uint8)
```

---

## 2. 評価指標

### 最終スコア（LB Score）

$$Score = \frac{SSIM + PSNR_{norm}}{2}$$

| 成分 | 計算式 | 値域 |
|------|--------|------|
| **SSIM** | 構造的類似度（mask内で計算） | 0.0 〜 1.0 |
| **PSNR** | $20 \times \log_{10}(255 / \sqrt{MSE})$ | dB単位 |
| **PSNR_norm** | $\text{clip}((PSNR - 15) / 20, 0, 1)$ | 0.0 〜 1.0 |

> [!IMPORTANT]
> - **PSNR 15 dB → norm 0.0**
> - **PSNR 35 dB → norm 1.0**
> - 評価は**マスク領域内**で行われる（data_range=255）

### SSIM 詳細

```
SSIM(x,y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
C1 = (0.01 × 255)², C2 = (0.03 × 255)²
```

### PSNR 詳細

```
PSNR = 20 × log10(255 / √MSE)  [dB]
```

---

## 3. 提出形式

| 項目 | 仕様 |
|------|------|
| **形式** | CSV |
| **カラム** | `id, pixel_0, pixel_1, ..., pixel_262143` |
| **ピクセル数** | 262,144 (= 512 × 512) |

### 提出ファイル例

```csv
id,pixel_0,pixel_1,pixel_2,...,pixel_262143
test_001,128,135,142,...,120
test_002,100,105,110,...,98
```

---

## 4. 実務上の重要ポイント
trainセットの透過画像にオルガノイドがうまく写っておらず、蛍光画像(target)の全画素が０である画像が四枚あります。

train_00099_target.png
train_00603_target.png
train_00802_target.png
train_00863_target.png
なお、testセットにそのような画像は含まれません。
### 優先順位

1. **構造保持が最優先（SSIM）**
   - エッジ・テクスチャの再現性が評価の要
   - ぼやけた出力は厳しくペナルティを受ける

2. **絶対輝度の安定性が重要**
   - sample間でmeanがブレない
   - 極端に暗く/明るくならない
   - 45閾値付近が潰れない

3. **ノイズはPSNRを著しく下げる**
   - ソルト&ペッパーノイズは致命的
   - スムージングと詳細保持のバランス

---

## 5. 検証済み知見（実験結果より）

> [!NOTE]
> 以下は実験結果に基づく **検証済みの知見**。

### ベストスコア推移

| 実験 | 変更内容 | LB Score | 累積改善 |
|------|----------|----------|----------|
| exp_015 | Baseline Fix | 0.407 | - |
| exp_016 | EfficientNet-b4 | 0.419 | +0.012 |
| exp_018 | TTA (4-way) | 0.428 | +0.009 |
| **exp_019** | **U-Net++** | **0.4307** | +0.003 |
| exp_022 | Augmentation | (実行中) | - |

**現在のベスト: 0.4307** / 目標: 0.46

### 効果あり ✅

| 手法 | 効果 | 実験 |
|------|------|------|
| **EfficientNet-b4** | +0.012 | exp_016 |
| **TTA (4-way flip)** | +0.009 | exp_018 |
| **U-Net++** | +0.003 | exp_019 |
| **OptimizedLoss** (L1+SSIM+Grad+TV) | 安定 | exp_017e |

### 効果なし/逆効果 ❌

| 手法 | 結果 | 実験 |
|------|------|------|
| Mean Matching | 効果なし (LB同等) | exp_014/015 |
| EdgeWeightedLoss (v1) | 不安定 | exp_020 |
| Temperature調整のみ | 微小効果 | exp_025/026 |

### 現在の最適構成

```python
# Model
architecture = "unetplusplus"
encoder = "efficientnet-b4"
encoder_weights = "imagenet"

# Loss
loss_type = "optimized"  # L1 + SSIM + Grad + TV
ssim_weight = 1.0
grad_weight = 0.5

# Inference
tta_enabled = True  # 4-way flip average

# exp_022 (testing)
augmentation_enabled = True
augmentation_strength = 0.5
```

---

## 6. 除外サンプル

以下4サンプルはターゲットが全黒のため学習から除外：

```python
EXCLUDED_SAMPLE_IDS = {
    'train_00099', 'train_00603', 
    'train_00802', 'train_00863'
}
```

---

## 7. 次の改善候補

1. **Augmentation** (exp_022) - 実行中
2. **Encoder upgrade** (B5/B6)
3. **3ch入力** (グレースケール+エッジ+コントラスト)
4. **補助ロス** (カテゴリ分類)
5. **Pseudo-labeling**

---

## 8. 制約事項

- [x] 評価はマスク領域内で計算 (data_range=255)
- [x] 提出形式: 512×512画像 → 262,144ピクセルフラット化
- [ ] 提出回数制限（要確認）
- [ ] 推論時間制限（要確認）

