#!/usr/bin/env python3
"""
Kaggle Training Trigger Script

This script can be used locally to trigger and monitor Kaggle training.
Requires: pip install kaggle
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_kaggle_command(args: list) -> tuple[int, str, str]:
    """Run a kaggle CLI command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["kaggle"] + args,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def push_kernel(kaggle_dir: Path, username: str) -> bool:
    """Push the kernel to Kaggle."""
    print("📤 Pushing kernel to Kaggle...")
    
    # Update metadata with username
    metadata_path = kaggle_dir / "kernel-metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    metadata["id"] = f"{username}/medical-ai-training"
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Push
    returncode, stdout, stderr = run_kaggle_command([
        "kernels", "push", "-p", str(kaggle_dir)
    ])
    
    if returncode != 0:
        print(f"❌ Failed to push kernel: {stderr}")
        return False
    
    print(f"✅ Kernel pushed successfully")
    print(stdout)
    return True


def wait_for_completion(kernel_id: str, max_wait: int = 21600, poll_interval: int = 300) -> str:
    """Wait for kernel to complete. Returns status."""
    print(f"⏳ Waiting for kernel completion: {kernel_id}")
    print(f"   Max wait: {max_wait/3600:.1f} hours, Poll interval: {poll_interval/60:.0f} minutes")
    
    elapsed = 0
    while elapsed < max_wait:
        returncode, stdout, stderr = run_kaggle_command([
            "kernels", "status", kernel_id
        ])
        
        # Parse status from output
        status = "unknown"
        if "complete" in stdout.lower():
            status = "complete"
        elif "running" in stdout.lower():
            status = "running"
        elif "error" in stdout.lower():
            status = "error"
        elif "queued" in stdout.lower():
            status = "queued"
        
        print(f"   [{elapsed//60:3d}m] Status: {status}")
        
        if status == "complete":
            print("✅ Kernel completed!")
            return "complete"
        elif status == "error":
            print("❌ Kernel failed!")
            return "error"
        
        time.sleep(poll_interval)
        elapsed += poll_interval
    
    print("⏰ Timeout waiting for kernel")
    return "timeout"


def download_output(kernel_id: str, output_dir: Path) -> bool:
    """Download kernel output files."""
    print(f"📥 Downloading output to {output_dir}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    returncode, stdout, stderr = run_kaggle_command([
        "kernels", "output", kernel_id, "-p", str(output_dir)
    ])
    
    if returncode != 0:
        print(f"❌ Failed to download output: {stderr}")
        return False
    
    print("✅ Output downloaded")
    
    # Show downloaded files
    for f in output_dir.iterdir():
        print(f"   📄 {f.name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Trigger Kaggle training")
    parser.add_argument("--username", required=True, help="Kaggle username")
    parser.add_argument("--poll-interval", type=int, default=300, help="Poll interval in seconds (default: 300)")
    parser.add_argument("--max-wait", type=int, default=21600, help="Max wait time in seconds (default: 21600)")
    parser.add_argument("--output-dir", default="kaggle_output", help="Output directory (default: kaggle_output)")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
    args = parser.parse_args()
    
    # Check kaggle CLI
    returncode, _, _ = run_kaggle_command(["--version"])
    if returncode != 0:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)
    
    # Find kaggle directory
    script_dir = Path(__file__).parent.parent
    kaggle_dir = script_dir / "kaggle"
    
    if not kaggle_dir.exists():
        print(f"❌ Kaggle directory not found: {kaggle_dir}")
        sys.exit(1)
    
    kernel_id = f"{args.username}/medical-ai-training"
    
    # Push kernel
    if not push_kernel(kaggle_dir, args.username):
        sys.exit(1)
    
    if args.no_wait:
        print(f"\n📋 Kernel ID: {kernel_id}")
        print("   Check status: kaggle kernels status " + kernel_id)
        return
    
    # Wait for completion
    status = wait_for_completion(kernel_id, args.max_wait, args.poll_interval)
    
    if status != "complete":
        sys.exit(1)
    
    # Download output
    output_dir = Path(args.output_dir)
    if not download_output(kernel_id, output_dir):
        sys.exit(1)
    
    # Show metrics
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        print("\n" + "="*50)
        print("📊 Training Results")
        print("="*50)
        print(f"   SSIM: {metrics['metrics']['ssim']:.4f}")
        print(f"   PSNR: {metrics['metrics']['psnr']:.2f} dB")
        print(f"   Time: {metrics['training_time_seconds']/60:.1f} minutes")
        print("="*50)


if __name__ == "__main__":
    main()
