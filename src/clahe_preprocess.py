"""
CLAHE Preprocessing Module for VirtualStaining
Applies Contrast Limited Adaptive Histogram Equalization for enhanced image quality
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    
    Args:
        image: Input grayscale image (H, W) or (H, W, 1), values 0-255 uint8
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: 8x8)
    
    Returns:
        CLAHE-enhanced image with same shape as input
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = image[:, :, 0]
    
    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    enhanced = clahe.apply(image)
    
    return enhanced


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int] = (512, 512),
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to specified size.
    
    Args:
        image: Input image
        size: Target size (width, height)
        interpolation: OpenCV interpolation method
    
    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=interpolation)


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    apply_clahe_enhancement: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    normalize: bool = True
) -> np.ndarray:
    """
    Full preprocessing pipeline: resize + CLAHE + normalize.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        apply_clahe_enhancement: Whether to apply CLAHE
        clip_limit: CLAHE clip limit
        tile_grid_size: CLAHE tile grid size
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Preprocessed image
    """
    # Resize
    image = resize_image(image, target_size)
    
    # Apply CLAHE
    if apply_clahe_enhancement:
        image = apply_clahe(image, clip_limit, tile_grid_size)
    
    # Normalize to [0, 1]
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image


def load_and_preprocess(
    image_path: Path,
    target_size: Tuple[int, int] = (512, 512),
    apply_clahe_enhancement: bool = True
) -> np.ndarray:
    """
    Load an image from disk and apply full preprocessing.
    
    Args:
        image_path: Path to image file
        target_size: Target size (width, height)
        apply_clahe_enhancement: Whether to apply CLAHE
    
    Returns:
        Preprocessed image as float32 in [0, 1] range
    """
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Apply preprocessing
    return preprocess_image(image, target_size, apply_clahe_enhancement)


def batch_preprocess_directory(
    input_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (512, 512),
    apply_clahe_enhancement: bool = True,
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
) -> int:
    """
    Preprocess all images in a directory and save to output directory.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for preprocessed images
        target_size: Target size (width, height)
        apply_clahe_enhancement: Whether to apply CLAHE
        extensions: Valid image file extensions
    
    Returns:
        Number of images processed
    """
    from tqdm import tqdm
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    count = 0
    for img_path in tqdm(image_files, desc="Preprocessing"):
        try:
            # Load and preprocess
            processed = load_and_preprocess(img_path, target_size, apply_clahe_enhancement)
            
            # Convert back to uint8 for saving
            processed_uint8 = (processed * 255).astype(np.uint8)
            
            # Save
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), processed_uint8)
            count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CLAHE Preprocessing for VirtualStaining")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Input directory")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=512, help="Target image size")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE")
    args = parser.parse_args()
    
    count = batch_preprocess_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        target_size=(args.size, args.size),
        apply_clahe_enhancement=not args.no_clahe
    )
    print(f"Processed {count} images")
