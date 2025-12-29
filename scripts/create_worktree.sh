#!/bin/bash
# create_worktree.sh - 実験用worktreeを作成するスクリプト
#
# Usage: ./scripts/create_worktree.sh <experiment_id>
# Example: ./scripts/create_worktree.sh exp_002_augment

set -e

# 引数チェック
if [ -z "$1" ]; then
    echo "Error: Experiment ID is required"
    echo "Usage: ./scripts/create_worktree.sh <experiment_id>"
    echo "Example: ./scripts/create_worktree.sh exp_002_augment"
    exit 1
fi

EXPERIMENT_ID=$1
REPO_ROOT=$(git rev-parse --show-toplevel)
PARENT_DIR=$(dirname "$REPO_ROOT")
WORKTREE_PATH="$PARENT_DIR/$EXPERIMENT_ID"

echo "=== Creating Worktree for $EXPERIMENT_ID ==="

# 1. ブランチが既に存在するかチェック
if git show-ref --verify --quiet "refs/heads/$EXPERIMENT_ID"; then
    echo "Error: Branch '$EXPERIMENT_ID' already exists"
    echo "Delete it first: git branch -D $EXPERIMENT_ID"
    exit 1
fi

# 2. 新しいブランチを作成
echo "[1/4] Creating branch '$EXPERIMENT_ID' from main..."
git checkout main
git pull origin main
git branch "$EXPERIMENT_ID"

# 3. Worktreeを作成
echo "[2/4] Creating worktree at '$WORKTREE_PATH'..."
git worktree add "$WORKTREE_PATH" "$EXPERIMENT_ID"

# 4. 実験ディレクトリを作成
EXPERIMENT_DIR="$WORKTREE_PATH/experiments/$EXPERIMENT_ID"
echo "[3/4] Creating experiment directory '$EXPERIMENT_DIR'..."
mkdir -p "$EXPERIMENT_DIR"

# 5. テンプレートファイルを配置
echo "[4/4] Creating template files..."

cat > "$EXPERIMENT_DIR/hypothesis.md" << 'EOF'
# 仮説

## 変更内容

[何を変更するか具体的に記載]

## 期待する効果

- SSIM: [期待する値]
- PSNR: [期待する値]

## 根拠

[なぜ改善すると考えるか]

## リスク

[悪化する可能性がある点]
EOF

cat > "$EXPERIMENT_DIR/config.yaml" << EOF
# Experiment Configuration

experiment:
  id: $EXPERIMENT_ID
  name: ""
  description: ""
  created_at: $(date -Iseconds)
  author: ""

# Model Configuration
model:
  architecture: unet
  encoder: resnet34

# Training Configuration
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001

# Data Configuration
data:
  augmentation:
    enabled: false
EOF

cat > "$EXPERIMENT_DIR/result.md" << 'EOF'
# 結果

## 実験ID

(experiment_id)

## CI結果

| 指標 | 値 | 標準偏差 |
|------|-----|----------|
| SSIM | - | - |
| PSNR | - dB | - |

CI Status: ⏳ 未実行

## 分析

### 良かった点
- 

### 悪かった点
- 

## 判断

- [ ] 採用
- [ ] 却下
- [ ] 保留

## 次のアクション

1. 
EOF

echo ""
echo "=== Worktree Created Successfully ==="
echo ""
echo "Worktree path: $WORKTREE_PATH"
echo "Experiment dir: $EXPERIMENT_DIR"
echo ""
echo "Next steps:"
echo "  1. cd $WORKTREE_PATH"
echo "  2. Edit experiments/$EXPERIMENT_ID/hypothesis.md"
echo "  3. Make your changes"
echo "  4. Run tests: ./scripts/run_tests_local.sh"
echo "  5. Commit and push: git push origin $EXPERIMENT_ID"
echo "  6. Create a Pull Request"
