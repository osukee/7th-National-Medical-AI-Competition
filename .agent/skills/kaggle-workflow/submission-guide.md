# Kaggle Competition Workflow Guide

Complete end-to-end workflow for Kaggle competitions from experiment start to submission.

## Complete Workflow

```
1. Update TODO.md
   ↓
2. Train Model with Trackio
   ↓
3. Evaluate & Log Results
   ↓
4. Prepare Submission Notebook
   ↓
5. Submit to Kaggle
   ↓
6. Document Results in TODO.md
```

## Phase 1: Pre-Training (TODO.md Update)

**MANDATORY**: Update @todo.md BEFORE starting any work.

```markdown
## 現在進行中

### 実験XX: [実験名] - 2025-12-17 14:30
- **状態**: 進行中
- **目的**: [この実験の目的]
- **仮説**: [検証したい仮説]
- **モデル**: ConvNeXt Small (timm)
- **ハイパーパラメータ**:
  - img_size: 224
  - batch_size: 32
  - learning_rate: 0.0001
  - epochs: 100
  - optimizer: AdamW
  - scheduler: CosineAnnealingLR
- **期待結果**: Val Acc > 0.85
- **関連ファイル**: `src/scripts/train_exp01.py`
```

## Phase 2: Training with Trackio

### Setup Training Script

```python
import trackio

# Initialize tracking
trackio.init(
    project="competition-name",
    name="exp01_baseline",
    config={
        "model": "convnext_small",
        "img_size": 224,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "epochs": 100,
    }
)

# Training loop with logging
for epoch in range(epochs):
    train_metrics = train_epoch(...)
    val_metrics = validate(...)

    trackio.log({
        "epoch": epoch,
        **train_metrics,
        **val_metrics,
        "learning_rate": get_lr(),
    })

trackio.finish()
```

### Monitor Training

```bash
# Launch dashboard in another terminal
trackio show --project "competition-name"

# Monitor GPU
watch -n 1 nvidia-smi
```

## Phase 3: Cross-Validation

### Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n=== Fold {fold + 1}/5 ===")

    # Initialize fold tracking
    trackio.init(
        project="competition-name",
        name=f"exp01_fold{fold}",
        config={**base_config, "fold": fold}
    )

    # Train model
    model = create_model()
    val_score = train_fold(model, train_idx, val_idx)

    # Log fold result
    fold_scores.append(val_score)
    trackio.log({f"fold{fold}/val_score": val_score})

    # Save model
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'val_score': val_score,
            'fold': fold,
        },
        f'models/exp01_fold{fold}.pth'
    )

    trackio.finish()

# Calculate average
avg_score = np.mean(fold_scores)
std_score = np.std(fold_scores)

print(f"\n=== Cross-Validation Results ===")
print(f"Average: {avg_score:.4f} ± {std_score:.4f}")
print(f"Fold scores: {fold_scores}")
```

### Group K-Fold (for grouped/time-series data)

```python
from sklearn.model_selection import GroupKFold

# Prevent data leakage in grouped data
gkf = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    # Same training logic as StratifiedKFold
    ...
```

## Phase 4: Post-Training Documentation

### Update TODO.md with Results

```markdown
## 最近完了したタスク

### 2025-12-17 20:45 - 実験01: ベースラインモデル
- **結果**:
  - 5-fold CV Average: 0.8523 ± 0.0124
  - Fold scores: [0.8645, 0.8512, 0.8398, 0.8623, 0.8437]
  - Training time: 2.5 hours
- **学び**:
  - ConvNeXt Smallは安定して高精度
  - Fold 2でスコアが低い→データ分布確認が必要
  - Mixed precision trainingで2x高速化
- **次のアクション**:
  - Fold 2の詳細分析
  - データ拡張強化（exp02）
  - ConvNeXt Baseで実験（exp03）
```

### Commit Experiment Results

```bash
git add experiments/exp01/ todo.md
git commit -m "exp: Record exp01 baseline results (CV=0.8523±0.0124)

5-fold cross-validation with ConvNeXt Small.
Identified potential issue in Fold 2 (score=0.8398).

🤖 Generated with Claude Code"
```

## Phase 5: Submission Notebook Preparation

### Critical Constraints

**Remember Kaggle limitations:**
1. NO albumentations (use torchvision.transforms)
2. NO external model imports (define in notebook)
3. weights_only=False required for torch.load()
4. Paths must use `/kaggle/input/...`

### Standard Submission Structure

```python
# ===== 1. Imports =====
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ===== 2. Configuration =====
class CFG:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 224
    batch_size = 32
    num_workers = 2
    n_folds = 5
    model_name = 'convnext_small'
    num_classes = 10

# ===== 3. Paths =====
DATA_DIR = Path('/kaggle/input/competition-name')
MODEL_DIR = Path('/kaggle/input/your-model-dataset')
OUTPUT_DIR = Path('/kaggle/working')

# ===== 4. Transform (torchvision ONLY) =====
test_transform = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== 5. Model Definition (COMPLETE in notebook) =====
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.model_name,
            pretrained=False,
            num_classes=CFG.num_classes
        )

    def forward(self, x):
        return self.backbone(x)

# ===== 6. Load Models =====
models = []
for fold in range(CFG.n_folds):
    model = ImageModel().to(CFG.device)
    checkpoint = torch.load(
        MODEL_DIR / f'exp01_fold{fold}.pth',
        map_location=CFG.device,
        weights_only=False  # REQUIRED
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)

print(f"Loaded {len(models)} models")

# ===== 7. Inference =====
test_df = pd.read_csv(DATA_DIR / 'test.csv')
predictions = []

with torch.no_grad():
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Load image
        img_path = DATA_DIR / 'test_images' / f"{row['id']}.jpg"
        image = Image.open(img_path).convert('RGB')
        image = test_transform(image).unsqueeze(0).to(CFG.device)

        # Ensemble predictions
        fold_preds = []
        for model in models:
            output = model(image)
            pred = torch.softmax(output, dim=1)
            fold_preds.append(pred.cpu().numpy())

        # Average predictions
        avg_pred = np.mean(fold_preds, axis=0)
        final_pred = np.argmax(avg_pred)
        predictions.append(final_pred)

# ===== 8. Create Submission =====
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': predictions
})

submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print(f"Submission saved: {len(submission)} predictions")
print(submission.head())
```

### Pre-Submission Checklist

Run `/kaggle:submit` to get complete checklist, or verify manually:

- [ ] NO albumentations
- [ ] torchvision.transforms used
- [ ] weights_only=False in torch.load()
- [ ] Model defined completely in notebook
- [ ] Paths use /kaggle/input/...
- [ ] submission.csv format correct
- [ ] "Restart & Run All" successful

## Phase 6: Submission

### Upload Model Weights to Kaggle

```bash
# 1. Create dataset on Kaggle
# 2. Upload model weights (*.pth files)
# 3. Note dataset name: "your-username/your-model-dataset"
```

### Submit Notebook

1. Click "Save Version"
2. Wait for notebook to run
3. Click "Submit to Competition"
4. Wait for scoring
5. Record score

### Document Submission

```markdown
## 最近完了したタスク

### 2025-12-17 22:30 - 実験01: 初回提出
- **結果**:
  - Public LB: 0.8456
  - Private LB: (未公開)
  - CV vs LB差: -0.0067
- **学び**:
  - CV/LBの相関良好
  - アンサンブルが効果的
- **次のアクション**:
  - データ拡張強化で改善狙う
  - 別モデルアーキテクチャ試行
```

## Common Workflows

### Rapid Iteration

```
1. Small experiment (10% data, 10 epochs)
   ↓
2. Quick evaluation
   ↓
3. If promising → Full experiment
   ↓
4. If not → Next idea
```

### Ablation Study

```
1. Baseline (exp01)
   ↓
2. +Data Augmentation (exp02)
   ↓
3. +Larger Model (exp03)
   ↓
4. +TTA (exp04)
   ↓
5. Best combination → Submit
```

### Ensemble Strategy

```
1. Train diverse models
   - Different architectures
   - Different augmentations
   - Different seeds
   ↓
2. Weighted averaging
   ↓
3. Submit ensemble
```

## Best Practices

### TODO.md Management

1. **Before ANY work**: Update TODO.md
2. **After experiment**: Document results immediately
3. **Daily review**: Check next steps
4. **Weekly summary**: Reflect on progress

### Experiment Tracking

1. **Always use trackio**: No exceptions
2. **Descriptive names**: exp01_baseline, exp02_heavy_aug
3. **Full config logging**: Every hyperparameter
4. **Save models**: All fold checkpoints

### Submission Strategy

1. **Validate locally first**: CV score as baseline
2. **Start simple**: Single model submission
3. **Iterate quickly**: Fast feedback loop
4. **Ensemble later**: After solid base models

### CV/LB Correlation

Monitor CV vs LB差:
- **< 0.01**: Good correlation, trust CV
- **0.01-0.03**: Acceptable, be cautious
- **> 0.03**: Poor correlation, investigate data leakage

## Troubleshooting

### CV/LB Mismatch

**Problem**: CV score high, LB score low

**Causes**:
- Data leakage in validation
- Train/test distribution shift
- Overfitting to validation set

**Solutions**:
- Use GroupKFold for grouped data
- Check for temporal leakage
- Reduce model complexity

### Submission Errors

**Problem**: Notebook fails on Kaggle

**Common causes**:
- albumentations usage
- External imports
- Missing weights_only=False
- Wrong paths

**Solution**: Run `/kaggle:submit` checklist

## Quick Reference

```bash
# Start experiment
# 1. Update @todo.md
# 2. Run training with trackio
uv run python src/scripts/train_exp01.py

# Monitor
trackio show --project "competition-name"

# After training
# 3. Update @todo.md with results
# 4. Commit experiment
git add experiments/exp01/ todo.md
git commit -m "exp: Record exp01 results (CV=X.XXXX)"

# Prepare submission
# 5. Create submission notebook
# 6. Run `/kaggle:submit` checklist
# 7. Submit to Kaggle
# 8. Document submission in @todo.md
```

Happy competing! 🏆
