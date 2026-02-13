# exp_030: Mega Improvement

> 実施日: 2026-02-13

## 仮説

exp_023 (0.44039) からの大幅改善を目指し、7つの改善を同時投入:

1. **バグ修正**: `transfer_learning_enabled=True` が無効なパスを参照 → `False`に
2. **Encoder upgrade**: efficientnet-b4 → b5 (より大きなfeature extractor)
3. **Epochs増加**: 15 → 25 (収束改善)
4. **Pseudo-Labeling**: テスト300枚をPhase1モデルで予測→Phase2 fine-tune
5. **Augmentation強化**: GridDistortion追加 + 強度0.5→0.6
6. **全5 Fold Ensemble**: top-3 → 全5 fold
7. **8-way TTA**: flip4 → dihedral8

## 変更点

| 設定 | exp_023 | exp_030 |
|------|---------|---------|
| transfer_learning_enabled | True (bug) | **False** |
| encoder | efficientnet-b4 | **efficientnet-b5** |
| epochs | 15 | **25** |
| pseudo_label_enabled | N/A | **True** |
| pseudo_label_epochs | N/A | **10** |
| augmentation_strength | 0.5 | **0.6** |
| GridDistortion | なし | **追加** |
| n_folds_ensemble | 3 | **5** |
| tta_mode | flip4 | **dihedral8** |
| fold_rank_weights | [1.0, 0.7, 0.4] | **[1.0, 0.9, 0.8, 0.7, 0.6]** |

## 結果

| Metric | exp_023 | exp_030 |
|--------|---------|---------|
| Public Score | 0.44039 | **TBD** |
| CV SSIM | TBD | TBD |

## 備考

- Phase 1 (25 epochs) → Phase 2 (10 epochs pseudo-label fine-tune) → Inference
- Pseudo-label weight = 0.5 (信頼度低いため)
- Phase 2 LR = Phase 1 LR × 0.3 (catastrophic forgetting防止)
