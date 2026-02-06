"""
Category Analysis Script for Medical AI Competition
Analyzes differences between categories A, B, C
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import statistics

def analyze_image_stats(image_path):
    """Get basic statistics from an image."""
    try:
        img = Image.open(image_path).convert('L')
        arr = np.array(img)
        return {
            'mean': arr.mean(),
            'std': arr.std(),
            'min': arr.min(),
            'max': arr.max(),
            'median': np.median(arr),
        }
    except Exception as e:
        return None

def main():
    data_dir = Path("medical-ai-contest-7th-2025")
    train_csv = data_dir / "train.csv"
    
    if not train_csv.exists():
        print(f"train.csv not found at {train_csv}")
        return
    
    df = pd.read_csv(train_csv)
    print(f"Total samples: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts().sort_index())
    
    # Sample analysis (first 10 per category for speed)
    results = {cat: {'input': [], 'target': []} for cat in ['A', 'B', 'C']}
    
    for cat in ['A', 'B', 'C']:
        cat_df = df[df['category'] == cat].head(30)  # Sample 30 per category
        
        for _, row in cat_df.iterrows():
            input_path = data_dir / row['input_path']
            target_path = data_dir / row['target_path']
            
            input_stats = analyze_image_stats(input_path)
            target_stats = analyze_image_stats(target_path)
            
            if input_stats and target_stats:
                results[cat]['input'].append(input_stats)
                results[cat]['target'].append(target_stats)
    
    print("\n" + "="*60)
    print("INPUT IMAGE STATISTICS (Brightfield/Transmission)")
    print("="*60)
    
    for cat in ['A', 'B', 'C']:
        if results[cat]['input']:
            means = [s['mean'] for s in results[cat]['input']]
            stds = [s['std'] for s in results[cat]['input']]
            print(f"\nCategory {cat} (n={len(means)}):")
            print(f"  Mean brightness: {np.mean(means):.2f} ± {np.std(means):.2f}")
            print(f"  Avg contrast (std): {np.mean(stds):.2f} ± {np.std(stds):.2f}")
    
    print("\n" + "="*60)
    print("TARGET IMAGE STATISTICS (Fluorescence)")
    print("="*60)
    
    for cat in ['A', 'B', 'C']:
        if results[cat]['target']:
            means = [s['mean'] for s in results[cat]['target']]
            stds = [s['std'] for s in results[cat]['target']]
            print(f"\nCategory {cat} (n={len(means)}):")
            print(f"  Mean brightness: {np.mean(means):.2f} ± {np.std(means):.2f}")
            print(f"  Avg contrast (std): {np.mean(stds):.2f} ± {np.std(stds):.2f}")
    
    print("\n" + "="*60)
    print("INPUT→TARGET DIFFERENCE (Complexity Indicator)")
    print("="*60)
    
    for cat in ['A', 'B', 'C']:
        if results[cat]['input'] and results[cat]['target']:
            input_means = [s['mean'] for s in results[cat]['input']]
            target_means = [s['mean'] for s in results[cat]['target']]
            diffs = [t - i for i, t in zip(input_means, target_means)]
            print(f"\nCategory {cat}:")
            print(f"  Brightness shift: {np.mean(diffs):.2f} ± {np.std(diffs):.2f}")
            print(f"  Range: [{min(diffs):.2f}, {max(diffs):.2f}]")

if __name__ == "__main__":
    main()
