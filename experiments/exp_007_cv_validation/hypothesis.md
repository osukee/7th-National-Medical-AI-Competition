# Experiment 007: Stratified K-Fold CV Validation

## Hypothesis

Stratified K-Fold Cross-Validation will provide:
1. **More reliable performance estimates** - CV reduces variance from single random split
2. **Category-wise insights** - Identify which category (A, B, or C) is the bottleneck
3. **Better model selection** - CV mean SSIM is more robust than single-fold SSIM

Based on prior analysis, we expect:
- Category A: Highest SSIM (high contrast, stable transformation)
- Category B: Mid-range SSIM
- Category C: Lowest SSIM (low contrast, high variance transformation)

## Changes from Baseline

| Parameter | exp_001 (Baseline) | exp_007 (This) |
|-----------|-------------------|----------------|
| Validation | Random 80/20 split | Stratified 5-Fold CV |
| Category metrics | None | A/B/C separate |
| ssim_weight | 1.0 | 2.0 (from exp_006) |
| Epochs | 3 | 5 per fold |

## Success Criteria

- [ ] CV runs successfully with 5 folds
- [ ] Category-wise SSIM/PSNR reported for each fold
- [ ] cv_results.json contains aggregated statistics
- [ ] Identify bottleneck category (expected: Category C)

## Expected Output Structure

```json
{
  "cv_results": {
    "n_folds": 5,
    "ssim_mean": 0.85,
    "ssim_std": 0.02,
    "category_metrics": {
      "A": {"ssim_mean": 0.88},
      "B": {"ssim_mean": 0.86},
      "C": {"ssim_mean": 0.81}
    }
  }
}
```

## Commands

```bash
# Environment variables for Kaggle
export N_FOLDS=5
export EPOCHS=5
python kaggle/train_notebook.py
```
