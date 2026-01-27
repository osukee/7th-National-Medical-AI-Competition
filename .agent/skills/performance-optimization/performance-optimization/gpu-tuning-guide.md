# GPU Performance Tuning Guide

Comprehensive guide for optimizing training performance in Kaggle competitions.

## GPU-Specific Recommendations

### Batch Size & Num Workers by GPU

| GPU | VRAM | Image Size | batch_size | num_workers | Notes |
|-----|------|------------|------------|-------------|-------|
| **RTX 4090** | 24GB | 512×512 | 16-24 | 8-12 | High-end consumer |
| **RTX 4090** | 24GB | 224×224 | 32-48 | 8-12 | Can push larger batches |
| **RTX 3090** | 24GB | 512×512 | 12-16 | 6-8 | Slightly slower than 4090 |
| **RTX 3090** | 24GB | 224×224 | 24-32 | 6-8 | Good for mid-size models |
| **RTX 3080** | 10GB | 512×512 | 6-8 | 4-6 | Memory-limited |
| **RTX 3080** | 10GB | 224×224 | 16-24 | 4-6 | Reduce batch if OOM |
| **T4 (Kaggle)** | 16GB | 512×512 | 8-12 | 2-4 | Kaggle default |
| **T4 (Kaggle)** | 16GB | 224×224 | 16-32 | 2-4 | Conservative num_workers |
| **V100** | 16GB | 512×512 | 12-16 | 4-8 | Fast compute, limited VRAM |
| **A100** | 40GB | 512×512 | 32-48 | 8-12 | Enterprise-grade |

### Finding Optimal Batch Size

```python
import torch

def find_max_batch_size(model, input_shape, device='cuda', start_batch=1):
    """Binary search for maximum batch size that fits in memory."""
    model.eval()
    model.to(device)

    low, high = start_batch, 1024
    max_batch = start_batch

    while low <= high:
        mid = (low + high) // 2
        try:
            # Test forward pass
            dummy_input = torch.randn(mid, *input_shape).to(device)
            with torch.no_grad():
                _ = model(dummy_input)

            # Success - try larger
            max_batch = mid
            low = mid + 1

            # Cleanup
            del dummy_input
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                # OOM - try smaller
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"Maximum batch size: {max_batch}")
    return max_batch

# Usage
model = YourModel()
max_bs = find_max_batch_size(
    model,
    input_shape=(3, 224, 224),  # C, H, W
    device='cuda'
)
```

## Mixed Precision Training

**Effect**: 2-3x speedup, 50% memory reduction

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Mixed Precision with Gradient Clipping

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
max_grad_norm = 1.0  # Gradient clipping threshold

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale loss and backward
        scaler.scale(loss).backward()

        # Unscale before gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Step with scaler
        scaler.step(optimizer)
        scaler.update()
```

## Gradient Accumulation

**Effect**: Simulate larger batch sizes without OOM

```python
accumulation_steps = 4  # Effective batch_size = batch_size * 4

model.train()
optimizer.zero_grad()

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every N steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining batches
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision + Gradient Accumulation

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
accumulation_steps = 4

model.train()
optimizer.zero_grad()

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()

        # Forward with autocast
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps

        # Backward with scaling
        scaler.scale(loss).backward()

        # Update every N steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

## DataLoader Optimization

```python
from torch.utils.data import DataLoader

# Optimized DataLoader settings
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # CPU cores for data loading
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True, # Keep workers alive (PyTorch 1.7+)
    prefetch_factor=2,      # Prefetch batches (PyTorch 1.7+)
)
```

### Finding Optimal num_workers

```python
import time
from torch.utils.data import DataLoader

def benchmark_dataloader(dataset, num_workers_list, batch_size=32):
    """Benchmark different num_workers settings."""
    results = {}

    for num_workers in num_workers_list:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

        # Warmup
        for _ in zip(range(10), loader):
            pass

        # Benchmark
        start = time.time()
        for _ in loader:
            pass
        elapsed = time.time() - start

        results[num_workers] = elapsed
        print(f"num_workers={num_workers}: {elapsed:.2f}s")

    best = min(results, key=results.get)
    print(f"\nBest num_workers: {best}")
    return best

# Usage
optimal_workers = benchmark_dataloader(
    train_dataset,
    num_workers_list=[0, 2, 4, 8, 12, 16]
)
```

## Memory Profiling

### Check GPU Memory Usage

```python
import torch

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Usage
print_gpu_memory()  # Before training
# ... training code ...
print_gpu_memory()  # After training
```

### Profile Memory During Training

```python
import torch

def profile_memory_usage(model, input_shape, device='cuda'):
    """Profile memory usage during forward/backward pass."""
    model.to(device)
    model.train()

    dummy_input = torch.randn(1, *input_shape).to(device)
    dummy_target = torch.randint(0, 10, (1,)).to(device)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Forward pass
    output = model(dummy_input)
    forward_mem = torch.cuda.max_memory_allocated() / 1024**3

    # Backward pass
    loss = torch.nn.functional.cross_entropy(output, dummy_target)
    loss.backward()
    backward_mem = torch.cuda.max_memory_allocated() / 1024**3

    print(f"Forward pass: {forward_mem:.2f}GB")
    print(f"Forward+Backward: {backward_mem:.2f}GB")

    # Cleanup
    del dummy_input, dummy_target, output, loss
    torch.cuda.empty_cache()

# Usage
profile_memory_usage(model, input_shape=(3, 224, 224))
```

## Parallel Processing (CPU)

### Process Data in Parallel

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

def process_item(item):
    """Process single item (CPU-heavy operation)."""
    # Your preprocessing logic here
    return processed_item

# Parallel processing
n_cores = multiprocessing.cpu_count()
max_workers = max(1, n_cores - 1)  # Leave 1 core for system

items = [...]  # Your data items

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    futures = {executor.submit(process_item, item): item for item in items}

    # Collect results with progress bar
    results = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())

print(f"Processed {len(results)} items using {max_workers} workers")
```

## Common Performance Issues

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Reduce `batch_size`
2. Use gradient accumulation
3. Use mixed precision training
4. Reduce image size
5. Use smaller model
6. Clear cache: `torch.cuda.empty_cache()`

### Issue 2: Slow Training

**Symptoms:**
- Low GPU utilization (< 80%)
- Training takes much longer than expected

**Solutions:**
1. Increase `num_workers` (but not > CPU cores)
2. Enable `pin_memory=True`
3. Use mixed precision training
4. Check data loading bottleneck
5. Use `persistent_workers=True`
6. Increase `prefetch_factor`

### Issue 3: CPU Bottleneck

**Symptoms:**
- GPU utilization drops periodically
- Data loading is slow

**Solutions:**
1. Increase `num_workers`
2. Simplify data augmentation
3. Pre-process data offline
4. Use faster image decoding (e.g., turbojpeg)
5. Cache preprocessed data in RAM

### Issue 4: Memory Leak

**Symptoms:**
- Memory usage grows over time
- OOM after many epochs

**Solutions:**
1. Detach tensors before logging: `loss.item()`
2. Don't accumulate history: use `loss.backward()` immediately
3. Clear cache periodically: `torch.cuda.empty_cache()`
4. Delete unused variables: `del unused_var`

## Performance Checklist

Before training:
- [ ] GPU memory checked with `nvidia-smi`
- [ ] Optimal `batch_size` determined
- [ ] Optimal `num_workers` benchmarked
- [ ] `pin_memory=True` enabled
- [ ] Mixed precision training configured

During training:
- [ ] GPU utilization > 80% (`nvidia-smi`)
- [ ] No memory warnings
- [ ] Training speed acceptable
- [ ] Metrics logged with trackio

After training:
- [ ] Performance metrics documented
- [ ] Optimal settings recorded in config

## Quick Reference

```python
# Optimal DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,          # Tune based on GPU
    shuffle=True,
    num_workers=4,          # Tune with benchmark
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

# Mixed Precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient Accumulation
accumulation_steps = 4
loss = loss / accumulation_steps
loss.backward()
if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

Happy optimizing! 🚀
