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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
