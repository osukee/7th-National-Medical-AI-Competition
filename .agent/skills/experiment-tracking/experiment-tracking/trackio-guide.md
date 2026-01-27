# Trackio Integration Guide

Complete guide for integrating Trackio experiment tracking in Kaggle competitions.

## Quick Start

```python
import trackio

# 1. Initialize
trackio.init(
    project="competition-name",
    config={
        "model": "convnext_small",
        "img_size": 224,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "epochs": 100,
    }
)

# 2. Log metrics in training loop
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)

    trackio.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr'],
    })

# 3. Finish
trackio.finish()
```

## Complete Training Script Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import trackio

# Configuration
config = {
    # Model
    "model_name": "convnext_small",
    "pretrained": True,
    "img_size": 224,
    "num_classes": 10,

    # Training
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.0001,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "scheduler": "CosineAnnealingLR",

    # Data
    "n_folds": 5,
    "train_split": 0.8,

    # Hardware
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
}

# Initialize tracking
trackio.init(
    project="kaggle-competition",
    name="exp01_baseline",
    config=config
)

# Model, optimizer, scheduler setup
model = create_model(config).to(config["device"])
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"]
)
criterion = nn.CrossEntropyLoss()

# Training loop
best_val_acc = 0.0

for epoch in range(config["epochs"]):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(config["device"]), labels.to(config["device"])

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100.0 * train_correct / train_total

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config["device"]), labels.to(config["device"])

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total

    # Update scheduler
    scheduler.step()

    # Log metrics
    trackio.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr'],
    })

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': config,
        }, 'models/best_model.pth')

        trackio.log({"best_val_acc": best_val_acc})

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")

# Finish tracking
trackio.finish()
print(f"Training complete! Best Val Acc: {best_val_acc:.2f}%")
```

## Cross-Validation Tracking

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== Fold {fold + 1}/5 ===")

    # Initialize fold-specific run
    trackio.init(
        project="kaggle-competition",
        name=f"exp01_fold{fold}",
        config={**config, "fold": fold}
    )

    # Prepare data
    train_dataset = create_dataset(X[train_idx], y[train_idx])
    val_dataset = create_dataset(X[val_idx], y[val_idx])

    # Train model
    model = create_model(config)
    best_val_acc = train_model(model, train_dataset, val_dataset)

    # Log fold result
    fold_scores.append(best_val_acc)
    trackio.log({
        f"fold{fold}/best_val_acc": best_val_acc,
    })

    trackio.finish()

# Log averaged results
avg_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print(f"\n=== Cross-Validation Results ===")
print(f"Average: {avg_score:.4f} ± {std_score:.4f}")
print(f"Fold scores: {fold_scores}")
```

## View Dashboard

After training starts, view metrics in real-time:

```bash
trackio show --project "kaggle-competition"
```

Browser opens at: http://127.0.0.1:7860

Dashboard features:
- Runs comparison table
- Interactive metric charts
- Hyperparameter comparison
- System metrics (GPU, memory)

## Advanced Logging

### Log Images

```python
import trackio

# Log sample predictions
sample_images = []
for img, pred, true in zip(images[:10], predictions[:10], labels[:10]):
    sample_images.append(
        trackio.Image(img, caption=f"Pred: {pred}, True: {true}")
    )

trackio.log({"sample_predictions": sample_images})
```

### Log Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')

# Log
trackio.log({"confusion_matrix": trackio.Image(plt)})
plt.close()
```

### Log Custom Metrics

```python
from sklearn.metrics import f1_score, precision_score, recall_score

# Calculate metrics
f1 = f1_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

# Log
trackio.log({
    "metrics/f1_score": f1,
    "metrics/precision": precision,
    "metrics/recall": recall,
})
```

## Best Practices

1. **Always initialize trackio at training start**
2. **Use hierarchical metric names** (e.g., "train/loss", "val/loss")
3. **Log hyperparameters in config** for easy comparison
4. **Track learning rate** to diagnose training issues
5. **Save best model** based on validation metrics
6. **Use descriptive project/run names**
7. **Log sample predictions** to visually verify model behavior

## Troubleshooting

### Dashboard not opening

```bash
# Check if trackio is running
ps aux | grep trackio

# Manually specify port
trackio show --project "kaggle-competition" --port 7861
```

### Database location

Trackio stores data in:
```
.trackio/trackio.db
```

This file is small (< 10MB) and can be committed to git.

## Integration Checklist

Before training:
- [ ] trackio.init() called with project name
- [ ] All hyperparameters in config dict
- [ ] Metric logging in training loop
- [ ] trackio.finish() at end of training

During training:
- [ ] Dashboard opened: `trackio show`
- [ ] Metrics updating in real-time
- [ ] Learning rate tracked

After training:
- [ ] Best model metrics logged
- [ ] Results recorded in @todo.md
- [ ] .trackio/ committed to git

Happy tracking! 📊
