"""
Category C Sub-Clustering Analysis
- Analyze C samples by brightness × contrast
- Identify difficulty clusters (easy/medium/hard)
- Output: train_with_difficulty.csv with 'difficulty' column

Run on Kaggle or locally with data available.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json


def analyze_image(image_path):
    """Extract brightness and contrast features from an image."""
    try:
        img = Image.open(image_path).convert('L')
        arr = np.array(img)
        return {
            'brightness': float(arr.mean()),
            'contrast': float(arr.std()),
            'dark_ratio': float((arr < 50).sum() / arr.size),  # % of very dark pixels
            'bright_ratio': float((arr > 200).sum() / arr.size),  # % of very bright pixels
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def cluster_category_c(df, data_dir, n_clusters=3):
    """
    Cluster Category C samples based on image features.
    Returns df with added 'difficulty' column.
    """
    print("="*60)
    print("Category C Sub-Clustering Analysis")
    print("="*60)
    
    # Filter to Category C only
    df_c = df[df['category'] == 'C'].copy()
    print(f"Category C samples: {len(df_c)}")
    
    # Extract features
    features = []
    valid_indices = []
    
    for idx, row in df_c.iterrows():
        input_path = Path(data_dir) / row['input_path']
        target_path = Path(data_dir) / row['target_path']
        
        input_stats = analyze_image(input_path)
        target_stats = analyze_image(target_path)
        
        if input_stats and target_stats:
            features.append([
                input_stats['brightness'],
                input_stats['contrast'],
                target_stats['contrast'],
                input_stats['dark_ratio'],
                abs(input_stats['brightness'] - target_stats['brightness']),  # transformation magnitude
            ])
            valid_indices.append(idx)
    
    features = np.array(features)
    print(f"Extracted features for {len(features)} samples")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters to determine difficulty
    cluster_stats = []
    for c in range(n_clusters):
        mask = clusters == c
        cluster_features = features[mask]
        cluster_stats.append({
            'cluster': c,
            'count': int(mask.sum()),
            'mean_input_brightness': float(cluster_features[:, 0].mean()),
            'mean_input_contrast': float(cluster_features[:, 1].mean()),
            'mean_target_contrast': float(cluster_features[:, 2].mean()),
            'mean_dark_ratio': float(cluster_features[:, 3].mean()),
            'mean_transform_magnitude': float(cluster_features[:, 4].mean()),
        })
    
    # Sort by difficulty (lower target contrast + higher dark ratio = harder)
    # Create a difficulty score: lower is easier
    for stat in cluster_stats:
        stat['difficulty_score'] = (
            -stat['mean_target_contrast'] +  # lower target contrast = harder
            stat['mean_dark_ratio'] * 100 +  # more dark pixels = harder
            stat['mean_transform_magnitude']  # larger transformation = harder
        )
    
    cluster_stats.sort(key=lambda x: x['difficulty_score'])
    
    # Map clusters to difficulty labels
    difficulty_map = {}
    difficulty_labels = ['C_easy', 'C_medium', 'C_hard'] if n_clusters == 3 else ['C_easy', 'C_hard']
    for i, stat in enumerate(cluster_stats):
        difficulty_map[stat['cluster']] = difficulty_labels[i]
    
    print("\nCluster Analysis:")
    print("-"*60)
    for stat in cluster_stats:
        label = difficulty_map[stat['cluster']]
        print(f"{label} (cluster {stat['cluster']}): {stat['count']} samples")
        print(f"  Input brightness: {stat['mean_input_brightness']:.2f}")
        print(f"  Input contrast:   {stat['mean_input_contrast']:.2f}")
        print(f"  Target contrast:  {stat['mean_target_contrast']:.2f}")
        print(f"  Dark ratio:       {stat['mean_dark_ratio']:.3f}")
        print(f"  Difficulty score: {stat['difficulty_score']:.2f}")
    
    # Assign difficulty labels to dataframe
    df['difficulty'] = df['category']  # Default: use category as difficulty
    for i, idx in enumerate(valid_indices):
        cluster_id = clusters[i]
        df.loc[idx, 'difficulty'] = difficulty_map[cluster_id]
    
    return df, cluster_stats, difficulty_map


def main():
    # Paths
    data_dir = Path(os.environ.get("DATA_DIR", "/kaggle/input/medical-ai-contest-7th-2025"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working"))
    train_csv = data_dir / "train.csv"
    
    if not train_csv.exists():
        print(f"train.csv not found at {train_csv}")
        print("Set DATA_DIR environment variable to point to data directory")
        return
    
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} samples")
    print(f"Category distribution: {df['category'].value_counts().to_dict()}")
    
    # Cluster Category C
    df_with_difficulty, cluster_stats, difficulty_map = cluster_category_c(df, data_dir, n_clusters=3)
    
    # Save results
    output_csv = output_dir / "train_with_difficulty.csv"
    df_with_difficulty.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    # Save cluster info
    cluster_info = {
        'cluster_stats': cluster_stats,
        'difficulty_map': {int(k): v for k, v in difficulty_map.items()},
        'difficulty_distribution': df_with_difficulty['difficulty'].value_counts().to_dict(),
    }
    
    with open(output_dir / "cluster_info.json", "w") as f:
        json.dump(cluster_info, f, indent=2)
    print(f"Saved: {output_dir / 'cluster_info.json'}")
    
    print("\n" + "="*60)
    print("Difficulty Distribution:")
    print("="*60)
    print(df_with_difficulty['difficulty'].value_counts().sort_index())


if __name__ == "__main__":
    main()
