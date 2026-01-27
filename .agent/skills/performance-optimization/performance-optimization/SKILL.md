---
name: performance-optimization
description: Optimize training performance for Kaggle competitions. Use PROACTIVELY when training is slow, GPU memory issues occur, or user asks about performance tuning. Keywords: 遅い, 最適化, OOM, CUDA, batch_size, num_workers, 高速化, パフォーマンス
---

## When to Use (PROACTIVE)

This skill should be activated automatically when:

- User reports slow training
- CUDA Out of Memory (OOM) errors occur
- User asks about batch_size or num_workers tuning
- User mentions: "遅い", "最適化", "OOM", "高速化"
- Training speed is suboptimal

## What This Skill Does

Provides GPU-specific optimization strategies:

- Batch size and num_workers recommendations by GPU
- Mixed Precision Training implementation
- Gradient Accumulation patterns
- Memory optimization techniques
- Parallel processing patterns

## How to Use

Refer to `gpu-tuning-guide.md` for:

- GPU-specific hyperparameter recommendations
- Mixed precision training code
- Gradient accumulation implementation
- Memory profiling techniques
- Common performance bottlenecks and solutions
