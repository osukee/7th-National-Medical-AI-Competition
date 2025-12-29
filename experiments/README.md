# Experiments ディレクトリ

## 運用ルール

> [!CAUTION]
> **仮説なき実験は禁止。結果を記録しない実験は「やっていない」と同義。**

---

## 命名規則

```
exp_XXX_description/
```

| 部分 | 説明 | 例 |
|------|------|-----|
| `exp_` | プレフィックス（固定） | - |
| `XXX` | 3桁の連番 | 001, 002, ..., 999 |
| `description` | 簡潔な説明（英語、スネークケース） | baseline, augment, unet_v2 |

---

## 必須ファイル

各実験ディレクトリには以下を **必ず** 含めること：

### 1. hypothesis.md

**実験開始前**に記載。

```markdown
# 仮説

## 変更内容
[何を変更するか]

## 期待する効果
[どの指標がどう改善するか]

## 根拠
[なぜ改善すると考えるか]

## リスク
[悪化する可能性がある点]
```

### 2. config.yaml

実験の再現に必要なすべての設定。

```yaml
experiment:
  id: exp_001_baseline
  created_at: 2024-01-01

model:
  architecture: unet
  encoder: resnet34

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001

data:
  augmentation: false
  train_split: 0.8
```

### 3. result.md

**実験終了後**に記載。CI結果に基づくこと。

```markdown
# 結果

## 実験ID
exp_001_baseline

## CI結果
- **ステータス**: Pass / Fail
- **SSIM**: 0.8500 ± 0.0100
- **PSNR**: 32.50 ± 0.50 dB

## 分析

### 良かった点
- [具体的な改善点]

### 悪かった点
- [具体的な問題点]

## 判断
採用 / 却下 / 保留

## 次のアクション
- [次に試すべき仮説]
```

---

## 実験一覧

| ID | 説明 | ステータス | SSIM | PSNR |
|----|------|------------|------|------|
| exp_001_baseline | ベースライン | 未実施 | - | - |

---

## 禁止事項

1. ❌ hypothesis.md なしで実験を開始
2. ❌ result.md なしで実験を終了
3. ❌ 複数の仮説を1実験で同時検証
4. ❌ CI結果を待たずに判断
5. ❌ 数値を手動計算して記載
