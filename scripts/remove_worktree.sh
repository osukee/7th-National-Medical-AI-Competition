#!/bin/bash
# remove_worktree.sh - 実験用worktreeを安全に削除するスクリプト
#
# Usage: ./scripts/remove_worktree.sh <experiment_id> [--force]
# Example: ./scripts/remove_worktree.sh exp_002_augment

set -e

# 引数チェック
if [ -z "$1" ]; then
    echo "Error: Experiment ID is required"
    echo "Usage: ./scripts/remove_worktree.sh <experiment_id> [--force]"
    echo "Example: ./scripts/remove_worktree.sh exp_002_augment"
    exit 1
fi

EXPERIMENT_ID=$1
FORCE_FLAG=$2
REPO_ROOT=$(git rev-parse --show-toplevel)
PARENT_DIR=$(dirname "$REPO_ROOT")
WORKTREE_PATH="$PARENT_DIR/$EXPERIMENT_ID"

echo "=== Removing Worktree for $EXPERIMENT_ID ==="

# 1. Worktreeが存在するかチェック
if [ ! -d "$WORKTREE_PATH" ]; then
    echo "Warning: Worktree directory does not exist: $WORKTREE_PATH"
else
    # 2. 未コミットの変更がないかチェック
    cd "$WORKTREE_PATH"
    if [ -n "$(git status --porcelain)" ]; then
        if [ "$FORCE_FLAG" != "--force" ]; then
            echo "Error: Worktree has uncommitted changes"
            echo "Commit your changes first, or use --force to discard them"
            git status --short
            exit 1
        else
            echo "Warning: Discarding uncommitted changes (--force specified)"
        fi
    fi
    cd "$REPO_ROOT"
fi

# 3. result.md が記入されているかチェック
RESULT_FILE="$WORKTREE_PATH/experiments/$EXPERIMENT_ID/result.md"
if [ -f "$RESULT_FILE" ]; then
    if grep -q "⏳ 未実行" "$RESULT_FILE"; then
        if [ "$FORCE_FLAG" != "--force" ]; then
            echo "Warning: result.md appears to be incomplete (CI not run)"
            echo "Complete the experiment record, or use --force to remove anyway"
            read -p "Continue anyway? (y/N): " confirm
            if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
                exit 1
            fi
        fi
    fi
fi

# 4. Worktreeを削除
echo "[1/3] Removing worktree..."
if [ -d "$WORKTREE_PATH" ]; then
    git worktree remove "$WORKTREE_PATH" --force 2>/dev/null || true
fi

# 5. Worktree参照をクリーンアップ
echo "[2/3] Cleaning up worktree references..."
git worktree prune

# 6. ローカルブランチを削除するか確認
echo "[3/3] Checking branch status..."
if git show-ref --verify --quiet "refs/heads/$EXPERIMENT_ID"; then
    read -p "Delete local branch '$EXPERIMENT_ID'? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        git branch -D "$EXPERIMENT_ID"
        echo "Local branch deleted"
    else
        echo "Local branch kept"
    fi
fi

echo ""
echo "=== Worktree Removed Successfully ==="
echo ""
echo "Note: Remote branch 'origin/$EXPERIMENT_ID' was NOT deleted."
echo "To delete remote branch: git push origin --delete $EXPERIMENT_ID"
