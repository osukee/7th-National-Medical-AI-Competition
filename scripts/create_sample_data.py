"""
Create sample training data for CI testing.
Generates minimal synthetic images for testing the training pipeline.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import csv

def create_sample_data(output_dir: str, num_samples: int = 5):
    """Create sample training data with synthetic images."""
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV data
    csv_rows = []
    
    for i in range(num_samples):
        sample_id = f"train_{i:05d}"
        
        # Create synthetic input image (grayscale, simulating transmission image)
        np.random.seed(i)
        input_img = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
        # Add some structure
        x, y = np.meshgrid(np.arange(512), np.arange(512))
        circle = ((x - 256) ** 2 + (y - 256) ** 2) < 100 ** 2
        input_img[circle] = np.clip(input_img[circle] + 50, 0, 255)
        
        # Create synthetic target image
        target_img = np.clip(input_img.astype(np.float32) * 0.8 + 30, 0, 255).astype(np.uint8)
        target_img = np.random.randint(30, 220, (512, 512), dtype=np.uint8)
        target_img[circle] = np.clip(target_img[circle] + 80, 0, 255)
        
        # Create mask image (binary)
        mask_img = np.zeros((512, 512), dtype=np.uint8)
        mask_img[circle] = 255
        
        # Save images
        Image.fromarray(input_img).save(train_dir / f"{sample_id}.png")
        Image.fromarray(target_img).save(train_dir / f"{sample_id}_target.png")
        Image.fromarray(mask_img).save(train_dir / f"{sample_id}_mask.png")
        
        csv_rows.append({
            "id": sample_id,
            "input_path": f"train/{sample_id}.png",
            "target_path": f"train/{sample_id}_target.png",
            "mask_path": f"train/{sample_id}_mask.png",
            "category": "A"
        })
        
        print(f"Created sample {sample_id}")
    
    # Write CSV
    csv_path = output_path / "train.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "input_path", "target_path", "mask_path", "category"])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"\nCreated {num_samples} samples in {output_path}")
    print(f"CSV written to {csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="sample-data")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()
    
    create_sample_data(args.output_dir, args.num_samples)
