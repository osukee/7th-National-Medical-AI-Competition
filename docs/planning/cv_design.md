# Cross-Validation設計ドキュメント

> [!IMPORTANT]
> このドキュメントはCV設計の根拠と実装方針を記載する。

---

## 1. データ概要

### 訓練データ
| 項目 | 値 |
|------|-----|
| サンプル数 | 1,200 |
| カテゴリ | A, B, C |
| カテゴリ分布 | **完全均等** (400:400:400) |

### テストデータ
| 項目 | 値 |
|------|-----|
| サンプル数 | 300 |
| カテゴリ情報 | **なし（非公開）** |

---

## 2. カテゴリ特性分析

### 2.1 INPUT画像（透過光画像）

| カテゴリ | 平均輝度 | コントラスト(std) | 特徴 |
|---------|---------|------------------|------|
| A | 78.91 ± 8.05 | 80.48 ± 9.38 | 明るく高コントラスト |
| B | 76.76 ± 8.33 | 80.69 ± 8.46 | Aに類似 |
| **C** | **66.46 ± 7.50** | **70.15 ± 8.31** | **暗く低コントラスト** |

### 2.2 TARGET画像（蛍光画像）

| カテゴリ | 平均輝度 | コントラスト(std) | 特徴 |
|---------|---------|------------------|------|
| **A** | 48.43 ± 6.07 | **92.36 ± 5.19** | **最も構造がはっきり** |
| B | 45.18 ± 7.14 | 82.78 ± 5.59 | 中間的 |
| **C** | 46.23 ± 10.28 | **66.71 ± 13.95** | **低コントラスト、ばらつき大** |

### 2.3 変換の複雑さ（INPUT→TARGET）

| カテゴリ | 輝度シフト | 範囲 | 解釈 |
|---------|-----------|------|------|
| A | -30.48 ± 12.88 | [-57, -2] | 安定した変換 |
| B | -31.59 ± 14.10 | [-60, +9] | やや不安定 |
| **C** | -20.24 ± 14.57 | [-61, +5] | **変換パターンが多様** |

---

## 3. 難易度評価

```
易 ←────────────────────→ 難
     A          B          C
```

| カテゴリ | 難易度 | 理由 |
|---------|--------|------|
| **A** | 🟢 易 | 高コントラスト、安定した変換パターン |
| **B** | 🟡 中 | 中間的な特性 |
| **C** | 🔴 難 | 低コントラスト、変換ばらつき大、予測困難 |

> [!WARNING]
> Category C の性能が全体スコアのボトルネックになる可能性が高い。

---

## 4. CV戦略

### 4.1 推奨手法: Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['category'])):
    train_df = df.iloc[train_idx]  # 960 samples
    val_df = df.iloc[val_idx]      # 240 samples
    # 各fold: A:320, B:320, C:320 (train) / A:80, B:80, C:80 (val)
```

### 4.2 なぜStratifiedか？

| 手法 | メリット | デメリット |
|------|---------|-----------|
| Random Split | シンプル | カテゴリ偏りのリスク |
| **Stratified K-Fold** | **カテゴリ比率維持** | やや複雑 |
| Group K-Fold | グループリーク防止 | カテゴリ = グループではない |

### 4.3 Fold設計

| Fold | Train | Validation |
|------|-------|------------|
| 各Fold共通 | A:320 + B:320 + C:320 = 960 | A:80 + B:80 + C:80 = 240 |

---

## 5. 実装チェックリスト

- [x] `train_notebook.py` を Stratified K-Fold に変更 → `train_kfold()` 関数追加
- [x] カテゴリ別SSIM/PSNR測定を追加 → `validate_with_categories()` 関数追加
- [x] 各foldの結果を記録・平均化 → `cv_results.json` として出力
- [ ] Category C の性能を重点監視 → 実験実行後に検証

---

## 6. 評価指標の拡張

### 実装済み: カテゴリ別メトリクス

```python
# kaggle/train_notebook.py の validate_with_categories() を参照
# 使用方法:
#   N_FOLDS=5 python kaggle/train_notebook.py
#
# 出力例:
#   Val SSIM: 0.8650 (A:0.89, B:0.87, C:0.83)
#   Val PSNR: 28.50 (A:30.2, B:28.8, C:26.5)
```

---

## 7. 実験履歴

### 7.1 exp_007 v1: 基本5-Fold CV

**実施日**: 2026-01-07

| Metric | Mean | Std |
|--------|------|-----|
| **SSIM** | 0.8662 | ±0.0183 |
| **PSNR** | 15.30 dB | ±1.76 |

**カテゴリ別SSIM**:
| A | B | C |
|---|---|---|
| 0.9126 | 0.8778 | 0.8081 |

> [!CAUTION]
> Category C はAより**0.10低い** → ボトルネック確定

---

### 7.2 exp_007 v2: Worst-Case CV追加

**実施日**: 2026-01-07

**新規KPI: Cの下位20% (ssim_C_worst20)**

| Metric | Value |
|--------|-------|
| ssim_C_worst20_mean | 0.7256 |
| **ssim_C_worst20_min** | **0.7047** |

**Fold別 C worst20%**:
| Fold | C_worst20 |
|------|-----------|
| 1 | 0.7155 |
| 2 | 0.7124 |
| 3 | 0.7634 |
| **4** | **0.7047** ← 最悪 |
| 5 | 0.7321 |

> [!WARNING]
> 平均SSIM 0.86 に対して worst20% は **0.70** → LBで刺される典型パターン

---

### 7.3 exp_007 v3: C サブクラスタリング

**実施日**: 2026-01-08

**変更内容**:
- Cを `C_easy`, `C_medium`, `C_hard` に分割（dark_ratio基準）
- K-Foldを5グループ (A, B, C_easy, C_medium, C_hard) で層化

**結果**: ⚠️ **リグレッション発生**

| Metric | v2 | v3 | 変化 |
|--------|-----|-----|------|
| ssim_mean | 0.8628 | 0.8714 | +0.009 ✅ |
| ssim_std | 0.0227 | 0.0187 | -0.004 ✅ |
| **ssim_C_worst20_min** | 0.7047 | **0.6544** | **-0.050** 🔴 |

**Fold別 C_worst20 (v3)**:
| Fold | C_worst20 |
|------|-----------|
| **1** | **0.6544** ← 大幅悪化 |
| 2 | 0.6730 |
| 3 | 0.7820 |
| 4 | 0.7472 |
| 5 | 0.7292 |

**分析**:
- 平均SSIMは改善、stdも改善 → 表面上は良く見える
- **worst-caseは0.05悪化** → C_hardが特定foldに集中した可能性
- Fold 1が異常に悪い → クラスタリングが意図通りに機能していない

> [!CAUTION]
> **教訓**: 平均・stdの改善だけでは判断できない。worst-caseを常に監視すべき。

---

### 7.4 exp_007 v4: Worst-Case Controlled CV

**実施日**: 2026-01-08

**設計変更** (v3の失敗から学習):

| 変更点 | v3 | v4 |
|--------|-----|-----|
| クラスタリング | 離散 (C_easy/medium/hard) | **連続値 (dark_ratio)** |
| C_hard の扱い | 全foldに分散 | **60%をTrain固定** |
| Worst評価 | 事後抽出 | **worst_val固定 (top 20%)** |
| KPI | ssim_C_worst20_min | **ssim_worst_val_min** |

**実装**:
```python
# 環境変数で切り替え (デフォルト: worst_case)
CV_MODE=worst_case python kaggle/train_notebook.py

# 旧モード (比較用)
CV_MODE=kfold python kaggle/train_notebook.py
```

**Split構造**:
```
Total: 1200 samples
├── worst_val (固定): 80 samples (C dark_ratio top 20%)
├── c_hard_train (固定): 96 samples (C_hard 60%)
└── trainable (K-Fold): 1024 samples
```

**結果**: 2026-01-09 実行完了

| Metric | v3 | v4 | 変化 |
|--------|-----|-----|------|
| ssim_mean | 0.8714 | 0.858 | -0.013 |
| ssim_std | 0.0187 | 0.033 | +0.014 |
| **ssim_worst_val_min** | 0.6544 | **0.461** | **-0.193** 🔴 |

**Fold別 worst_val 結果**:

| Fold | worst_val_mean | worst_val_min | 評価 |
|------|----------------|---------------|------|
| 1 | 0.711 | 0.487 | 中 |
| 2 | 0.695 | 0.496 | 悪 |
| **3** | **0.845** | **0.601** | **良** |
| 4 | 0.712 | **0.461** | **最悪** |
| 5 | 0.821 | 0.560 | 良 |

**カテゴリ別SSIM**:
| A | B | C |
|---|---|---|
| 0.903 | 0.871 | 0.801 |

> [!IMPORTANT]
> **v4の数値悪化は「失敗」ではない。真の最悪ケースが可視化された。**
> v3のworst20%は事後抽出だったため、本当のハードケースが隠れていた。

**分析**:
- Fold 3/5 は良好、Fold 2/4 は不安定 → モデル収束不足の可能性
- epochs=3 は不十分、10以上必要
- ssim_min=0.46 のサンプルが存在 → C向けLoss重み増加が必要

---

### 7.5 exp_007 v5: Worst-Val Training Integration

**実施日**: 2026-01-09

**設計変更** (v4の問題点から):

| 変更点 | v4 | v5 |
|--------|-----|-----|
| worst_val | 評価専用 (80) | **半分をTrain (40)** |
| Loss重み | 全sample同一 | **worst: ×3, C_hard: ×2** |
| 評価対象 | worst_val全体 | **worst_eval (10-20%)** |

**結果**: 🎉 **大幅改善**

| Metric | v4 | v5 | 変化 |
|--------|-----|-----|------|
| ssim_mean | 0.858 | **0.882** | **+0.024** ✅ |
| ssim_std | 0.033 | **0.013** | **-0.020** ✅ |
| ssim_worst_eval_min | 0.461 | **0.671** | **+0.210** 🎉 |

**Fold別 worst_eval 結果**:

| Fold | worst_eval_mean | worst_eval_min |
|------|-----------------|----------------|
| 1 | 0.840 | 0.713 |
| 2 | 0.800 | 0.673 |
| 3 | 0.839 | 0.671 |
| 4 | 0.843 | 0.683 |
| 5 | 0.834 | **0.732** |

**カテゴリ別SSIM**:
| A | B | C |
|---|---|---|
| 0.905 | 0.890 | **0.850** |

> [!TIP]
> **成功の鍵**: worst_train をLoss×3で学習させたことで、モデルがハードケースに勾配を向けた

**学び**:
- worst-case は「評価」だけでなく「学習」に使うべき
- epochs=3 でもこの改善 → Loss重みの効果は絶大
- 次はepochs増加でさらなる改善を検証

---

## 8. 学んだ教訓

### 8.1 CV設計の盲点

| 盲点 | 現象 | 対策 |
|------|------|------|
| 平均依存 | 平均SSIM改善でもworstが悪化 | **ssim_C_worst20_minをKPIに** |
| 単純層化 | Stratifiedでも運ゲー | Multi-seed CVで最悪fold検証 |
| クラスタリングの罠 | C分割でworstが悪化 | 分割前後でworst比較必須 |

### 8.2 競技向けCV設計原則

1. **平均を見るな、最悪を見ろ**
2. **改善は worst-case で測れ**
3. **実験前後で worst-case を必ず比較**

---

## 9. 追加検討項目

### 9.1 TTA (Test Time Augmentation)

```python
tta_transforms = [
    lambda x: x,                    # Original
    lambda x: torch.flip(x, [2]),   # Horizontal flip
    lambda x: torch.flip(x, [3]),   # Vertical flip
    lambda x: torch.rot90(x, 1, [2,3]),  # 90° rotation
]
```

**期待効果**: SSIM +0.01〜0.02

### 9.2 Ensemble戦略

| 戦略 | 期待効果 |
|------|----------|
| Fold Ensemble | +0.02〜0.03 |
| Architecture Ensemble | +0.03〜0.05 |

### 9.3 Category C 強化策

| 戦略 | 期待効果 |
|------|----------|
| C向けLoss重み増加 | +0.01〜0.02 |
| C専用Fine-tuning | +0.02〜0.03 |

---

## 10. 次のアクション

- [x] Stratified K-Fold実装
- [x] カテゴリ別評価追加
- [x] Worst-Case CV (ssim_C_worst20) 追加
- [x] C サブクラスタリング実験 → **リグレッション、要再検討**
- [ ] クラスタリング無効化して baseline 再取得
- [ ] Multi-Seed CV 検証
- [ ] C向けLoss重み調整
- [ ] TTA実装

---

*最終更新: 2026-01-08 (exp_007 v3 結果反映)*
