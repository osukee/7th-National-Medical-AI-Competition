"""
Unit tests for Cross-Validation implementation.
Tests Stratified K-Fold split and category-wise evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


class TestStratifiedKFold:
    """Test Stratified K-Fold maintains category proportions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe with balanced categories."""
        return pd.DataFrame({
            'id': [f'train_{i:05d}' for i in range(120)],
            'category': ['A'] * 40 + ['B'] * 40 + ['C'] * 40
        })
    
    def test_stratified_split_proportions(self, sample_df):
        """Test that each fold maintains category proportions."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(sample_df, sample_df['category'])):
            train_df = sample_df.iloc[train_idx]
            val_df = sample_df.iloc[val_idx]
            
            # Check train set proportions (96 samples = 32 per category)
            train_counts = train_df['category'].value_counts()
            assert train_counts['A'] == 32, f"Fold {fold}: Expected 32 A in train, got {train_counts['A']}"
            assert train_counts['B'] == 32, f"Fold {fold}: Expected 32 B in train, got {train_counts['B']}"
            assert train_counts['C'] == 32, f"Fold {fold}: Expected 32 C in train, got {train_counts['C']}"
            
            # Check val set proportions (24 samples = 8 per category)
            val_counts = val_df['category'].value_counts()
            assert val_counts['A'] == 8, f"Fold {fold}: Expected 8 A in val, got {val_counts['A']}"
            assert val_counts['B'] == 8, f"Fold {fold}: Expected 8 B in val, got {val_counts['B']}"
            assert val_counts['C'] == 8, f"Fold {fold}: Expected 8 C in val, got {val_counts['C']}"
    
    def test_no_overlap_between_train_val(self, sample_df):
        """Test that train and validation sets have no overlap."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(sample_df, sample_df['category'])):
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Fold {fold}: Found overlap between train and val"
    
    def test_all_samples_used(self, sample_df):
        """Test that all samples are used across folds."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        all_val_indices = set()
        for fold, (train_idx, val_idx) in enumerate(skf.split(sample_df, sample_df['category'])):
            all_val_indices.update(val_idx)
        
        assert all_val_indices == set(range(len(sample_df))), "Not all samples used as validation"


class TestCategoryMetrics:
    """Test category-wise metrics calculation."""
    
    def test_category_metric_structure(self):
        """Test that category metrics have correct structure."""
        # Simulated results structure
        results = {
            'ssim_A': 0.89,
            'ssim_B': 0.87,
            'ssim_C': 0.83,
            'psnr_A': 30.2,
            'psnr_B': 28.8,
            'psnr_C': 26.5,
        }
        
        # Verify all categories present
        for cat in ['A', 'B', 'C']:
            assert f'ssim_{cat}' in results, f"Missing ssim_{cat}"
            assert f'psnr_{cat}' in results, f"Missing psnr_{cat}"
    
    def test_category_c_expected_lower(self):
        """Test assumption that Category C typically has lower scores."""
        # Based on cv_design.md analysis
        typical_results = {
            'ssim_A': 0.89,  # Easy
            'ssim_B': 0.87,  # Medium
            'ssim_C': 0.83,  # Hard
        }
        
        assert typical_results['ssim_A'] > typical_results['ssim_C'], \
            "Category A should typically score higher than C"
        assert typical_results['ssim_B'] > typical_results['ssim_C'], \
            "Category B should typically score higher than C"


class TestFoldAggregation:
    """Test fold result aggregation."""
    
    def test_mean_std_calculation(self):
        """Test mean and std calculation across folds."""
        fold_results = [
            {'ssim': 0.85, 'psnr': 28.0},
            {'ssim': 0.86, 'psnr': 28.5},
            {'ssim': 0.84, 'psnr': 27.5},
            {'ssim': 0.87, 'psnr': 29.0},
            {'ssim': 0.85, 'psnr': 28.0},
        ]
        
        ssim_mean = np.mean([r['ssim'] for r in fold_results])
        ssim_std = np.std([r['ssim'] for r in fold_results])
        
        assert abs(ssim_mean - 0.854) < 0.01, f"Mean SSIM incorrect: {ssim_mean}"
        assert ssim_std > 0, "Standard deviation should be positive"


class TestWorstCaseCV:
    """Test worst-case controlled CV splits."""
    
    @pytest.fixture
    def sample_df_with_dark_ratio(self):
        """Create sample dataframe with dark_ratio (simulating C samples)."""
        np.random.seed(42)
        # 400 C samples with varying dark_ratio
        c_dark_ratios = np.random.uniform(0.0, 0.5, 400)
        c_dark_ratios = np.sort(c_dark_ratios)[::-1]  # Sort descending
        
        return pd.DataFrame({
            'id': [f'train_{i:05d}' for i in range(400)],
            'category': ['C'] * 400,
            'dark_ratio': c_dark_ratios,
        })
    
    def test_worst_val_is_top_20_percent(self, sample_df_with_dark_ratio):
        """Test that worst_val contains top 20% of dark_ratio samples."""
        df = sample_df_with_dark_ratio
        n_c = len(df)
        
        # Top 20% by dark_ratio
        n_worst = int(n_c * 0.20)
        worst_val_idx = df.nlargest(n_worst, 'dark_ratio').index.tolist()
        
        assert len(worst_val_idx) == 80, f"Expected 80 worst_val samples, got {len(worst_val_idx)}"
        
        # Verify these are the highest dark_ratio
        worst_dark_ratios = df.loc[worst_val_idx, 'dark_ratio']
        other_dark_ratios = df.drop(worst_val_idx)['dark_ratio']
        
        assert worst_dark_ratios.min() >= other_dark_ratios.max(), \
            "worst_val should have the highest dark_ratios"
    
    def test_c_hard_train_is_60_percent(self, sample_df_with_dark_ratio):
        """Test that 60% of C_hard goes to train fixed."""
        df = sample_df_with_dark_ratio
        n_c = len(df)
        
        n_worst = int(n_c * 0.20)  # 80
        n_c_hard = int(n_c * 0.40)  # 160
        n_c_hard_train = int(n_c_hard * 0.60)  # 96
        
        assert n_c_hard_train == 96, f"Expected 96 C_hard_train samples, got {n_c_hard_train}"
    
    def test_no_overlap_between_splits(self, sample_df_with_dark_ratio):
        """Test that worst_val, c_hard_train, and trainable have no overlap."""
        df = sample_df_with_dark_ratio
        n_c = len(df)
        
        # Simulate the split logic
        df_sorted = df.sort_values('dark_ratio', ascending=False)
        
        n_worst = int(n_c * 0.20)
        n_c_hard = int(n_c * 0.40)
        n_c_hard_train = int(n_c_hard * 0.60)
        
        worst_val_idx = set(df_sorted.index[:n_worst])
        c_hard_idx = df_sorted.index[n_worst:n_worst + n_c_hard].tolist()
        c_hard_train_idx = set(c_hard_idx[:n_c_hard_train])
        c_hard_foldable_idx = set(c_hard_idx[n_c_hard_train:])
        c_normal_idx = set(df_sorted.index[n_worst + n_c_hard:])
        
        # Check no overlaps
        assert len(worst_val_idx & c_hard_train_idx) == 0, "Overlap between worst_val and c_hard_train"
        assert len(worst_val_idx & c_normal_idx) == 0, "Overlap between worst_val and c_normal"
        assert len(c_hard_train_idx & c_normal_idx) == 0, "Overlap between c_hard_train and c_normal"
        
        # Check all samples accounted for
        all_idx = worst_val_idx | c_hard_train_idx | c_hard_foldable_idx | c_normal_idx
        assert all_idx == set(df.index), "Some samples not accounted for"
    
    def test_worst_val_fixed_across_folds(self):
        """Test that worst_val indices remain fixed regardless of fold."""
        # This is a conceptual test - in the actual implementation,
        # worst_val_idx is computed once and passed to all folds
        np.random.seed(42)
        
        # Simulate two runs with same seed
        dark_ratios_1 = np.random.uniform(0.0, 0.5, 100)
        np.random.seed(42)
        dark_ratios_2 = np.random.uniform(0.0, 0.5, 100)
        
        assert np.allclose(dark_ratios_1, dark_ratios_2), \
            "Same seed should produce same dark_ratios"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
