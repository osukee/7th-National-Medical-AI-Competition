"""
Figshare Dataset Downloader
Downloads VirtualStaining dataset from Figshare (DOI: 10.6084/m9.figshare.21971558)
"""

import argparse
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


FIGSHARE_API_BASE = "https://api.figshare.com/v2"
ARTICLE_ID = "21971558"  # From DOI: 10.6084/m9.figshare.21971558


def get_article_files(article_id: str) -> list:
    """Fetch file metadata from Figshare article."""
    url = f"{FIGSHARE_API_BASE}/articles/{article_id}/files"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract ZIP file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def download_virtualstaining(output_dir: Path, dry_run: bool = False) -> None:
    """Download entire VirtualStaining dataset from Figshare."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching file list from Figshare article {ARTICLE_ID}...")
    files = get_article_files(ARTICLE_ID)
    
    print(f"Found {len(files)} file(s):")
    for f in files:
        size_mb = f.get('size', 0) / (1024 * 1024)
        print(f"  - {f['name']} ({size_mb:.2f} MB)")
    
    if dry_run:
        print("\n[DRY RUN] Would download the above files.")
        return
    
    # Download each file
    for file_info in files:
        file_name = file_info['name']
        download_url = file_info['download_url']
        file_path = output_dir / file_name
        
        if file_path.exists():
            print(f"Skipping {file_name} (already exists)")
            continue
        
        print(f"\nDownloading {file_name}...")
        download_file(download_url, file_path)
        
        # Extract if ZIP
        if file_name.endswith('.zip'):
            extract_dir = output_dir / file_name.replace('.zip', '')
            extract_zip(file_path, extract_dir)
    
    print("\n✓ Download complete!")
    print(f"Dataset saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download VirtualStaining dataset from Figshare")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/VirtualStaining",
        help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be downloaded without actually downloading"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    download_virtualstaining(output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
