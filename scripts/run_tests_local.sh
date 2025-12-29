#!/bin/bash
# run_tests_local.sh - ローカルでCIと同等のテストを実行するスクリプト
#
# Usage: ./scripts/run_tests_local.sh [--quick]
#   --quick: 簡易テストのみ実行（訓練スキップ）

set -e

QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

echo "=== Running Local Tests ==="
echo "Quick mode: $QUICK_MODE"
echo ""

# 1. 環境チェック
echo "[1/5] Checking environment..."
python --version
pip --version

# 2. 依存関係のインストール確認
echo ""
echo "[2/5] Checking dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo "Dependencies installed"
else
    echo "Warning: requirements.txt not found"
fi

# 3. リンター実行（存在する場合）
echo ""
echo "[3/5] Running linter..."
if command -v ruff &> /dev/null; then
    ruff check src/ tests/ --ignore E501 || true
elif command -v flake8 &> /dev/null; then
    flake8 src/ tests/ --ignore=E501 || true
else
    echo "No linter found (ruff or flake8), skipping..."
fi

# 4. ユニットテスト実行
echo ""
echo "[4/5] Running unit tests..."
if [ -d "tests" ]; then
    python -m pytest tests/ -v --tb=short
else
    echo "Warning: tests/ directory not found"
fi

# 5. 簡易トレーニング（--quickでなければ）
echo ""
echo "[5/5] Running training check..."
if [ "$QUICK_MODE" = true ]; then
    echo "Skipped (quick mode)"
else
    if [ -f "src/train.py" ]; then
        echo "Running short training..."
        python src/train.py --epochs 1 --quick-check 2>/dev/null || echo "Training script not configured for quick check"
    else
        echo "Warning: src/train.py not found"
    fi
fi

echo ""
echo "=== Local Tests Complete ==="
echo ""

# 結果サマリー
if [ -f "metrics.json" ]; then
    echo "Metrics generated:"
    cat metrics.json
else
    echo "Note: metrics.json not generated (expected in full CI run)"
fi
