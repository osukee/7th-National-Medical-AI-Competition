# exp_013: Distribution Analysis

> 作成日: 2026-01-12

## 目的

LB最適化のための系統的アプローチ Step 1:
- LB評価関数を完全再現
- 出力分布を可視化
- 分布安定性をチェック → 次の手を決定

## 変更内容

| 項目 | 設定 |
|------|------|
| Model | resnet34, 512 (baseline) |
| 新機能 | `analyze_distribution = True` |

## 分析項目

- Pred vs Target の mean/std
- Sample間のばらつき
- Mean偏移 / Std比

## 診断閾値

| 項目 | 閾値 | 次の手 |
|------|------|--------|
| Mean偏移 | >10 | 分布正規化 (mean matching) |
| Std比 | <0.8 or >1.2 | コントラスト調整 |
| Sample間variance | >20 | 分布安定化 |

## 結果

> ⏳ Kaggle実行待ち...

### Distribution Analysis
```
(Kaggle出力から貼り付け)
```
