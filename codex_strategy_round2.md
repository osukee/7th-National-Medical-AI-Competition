# codex strategy round2（追加分析）

`codex_strategy_review.md` と `kaggle/train_notebook.py` を読み、要望4点を実装目線で追記する。

## (1) SWAを`train_kfold`へ組み込む具体コード案

方針: 最小差分で `train_kfold` に導入する。ポイントは以下。

1. 終盤エポックのみ `AveragedModel` を更新。
2. 終了時に `update_bn()` を必ず実行。
3. 「最終epoch」ではなく「best checkpoint基準」で fold 指標を確定。

### 追加import/Config

```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

class Config:
    ...
    swa_enabled = True
    swa_start_ratio = 0.7
    swa_lr = 5e-5
    swa_anneal_epochs = 3
```

### `train_kfold`差し替えイメージ（要点のみ）

```python
def train_kfold(config, n_folds=5):
    ...
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['difficulty'])):
        ...
        model = create_model(config)
        criterion = create_loss(config).to(config.device)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

        use_swa = getattr(config, 'swa_enabled', False)
        swa_start = int(config.epochs * getattr(config, 'swa_start_ratio', 0.7))
        swa_start = max(1, min(swa_start, config.epochs - 1))

        if use_swa:
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(
                optimizer,
                swa_lr=getattr(config, 'swa_lr', config.learning_rate * 0.5),
                anneal_strategy='cos',
                anneal_epochs=getattr(config, 'swa_anneal_epochs', 3),
            )

        best_fold_ssim = -1.0

        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
            val_metrics = validate_with_categories(model, val_loader, criterion, config.device, config)

            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            if val_metrics['ssim'] > best_fold_ssim:
                best_fold_ssim = val_metrics['ssim']
                torch.save(model.state_dict(), config.output_dir / f"best_model_fold{fold}.pth")

        # fold終端でSWAモデルを評価（BN再計算必須）
        if use_swa:
            update_bn(train_loader, swa_model, device=config.device)
            swa_metrics = validate_with_categories(swa_model.module, val_loader, criterion, config.device, config)
            if swa_metrics['ssim'] > best_fold_ssim:
                best_fold_ssim = swa_metrics['ssim']
                torch.save(swa_model.module.state_dict(), config.output_dir / f"best_model_fold{fold}.pth")

        # 重要: fold結果はbest checkpointを再ロードして算出
        best_state = torch.load(config.output_dir / f"best_model_fold{fold}.pth", map_location=config.device)
        model.load_state_dict(best_state)
        final_metrics = validate_with_categories(model, val_loader, criterion, config.device, config)
        ...
```

注意:

1. 本番デフォルトは `CV_MODE=worst_case_v5`（`train_kfold`だけ改修しても未適用）。
2. 同じSWAロジックを `train_worst_case_cv_v5` 側にも展開する必要あり。

## (2) Progressive Training 256→512 の検討

結論: 6時間制約では `512→576` より `256→512` の方が再現性と試行回数の期待値が高い。

理由:

1. 576は改善余地がある一方、計算コスト増が大きく探索本数が減る。
2. 提出時は最終的に 512 グリッドで扱うため、最終学習解像度を512に揃える方が推論条件と整合。
3. 256段階で大域構造を素早く学習し、512段階で境界を詰めるのが時間対効果に優れる。

推奨レシピ:

1. Phase1: `image_size=256`, 8-10 epochs, LR `1e-4`。
2. Phase2: `image_size=512`, 12-16 epochs, LR `5e-5`。
3. Optimizer stateは引継ぎ、schedulerはphaseごと再初期化。
4. SWAを使うなら Phase2 後半のみ適用。

実験時の注意:

1. 256を長く回し過ぎると高周波復元が遅れるため、`Phase2 >= Phase1` を維持。
2. 比較は「同一総GPU時間」で実施（公平比較）。

## (3) `train_notebook.py` バグ・非効率コードレビュー

### High

1. fold集計がbest checkpoint基準ではない。
- `train_kfold`で`best_model_fold{fold}.pth`を保存しているが、`fold_results`は最終epochモデルで評価している。
- 参照: `kaggle/train_notebook.py` 2023行付近（best保存）、2027-2040行付近（最終モデル評価）。
- 影響: fold rankingが汚れ、top-3 fold選抜の精度が落ちる。

2. overall best (`best_model.pth`) 判定も最終epochモデル基準。
- 参照: `kaggle/train_notebook.py` 2052-2056行付近。
- 影響: 実際のbest foldを取り逃す可能性。

3. `train_epoch_weighted` の加重平均式が不正確。
- 現状は `mean(w_i * loss_i)` で、`sum(w_i*loss_i)/sum(w_i)` ではない。
- 参照: `kaggle/train_notebook.py` 1603行付近。
- 影響: バッチごとのweight総和でlossスケールが変動。

### Medium

1. 全体SSIMがカテゴリ単純平均。
- 現状はA/B/Cの平均で、サンプル数重みを反映しない。
- 参照: `kaggle/train_notebook.py` 1750-1755行付近。
- 影響: 実サンプル分布に対してCV指標が歪む。

2. `calculate_ssim_masked` が局所窓SSIMではなく、マスク領域統計量ベース実装。
- 参照: `kaggle/train_notebook.py` 1335-1364行。
- 影響: `skimage.metrics.structural_similarity` と性質が異なり、CV/LB整合にズレ余地。

3. 出力への`clamp`が学習時に常時適用。
- モデル側が既にsigmoid出力。
- 参照: `kaggle/train_notebook.py` 731行（sigmoid）, 1549/1590行（clamp）。
- 影響: 冗長。0/1近傍で勾配の情報量を減らす懸念。

4. `predict_and_submit`で`weights_tensor`を毎バッチ再生成。
- 参照: `kaggle/train_notebook.py` 3194-3195行。
- 影響: 小さいが無駄なGPU/CPUオーバーヘッド。

### Low

1. `OrganoidDataset` が `Config` クラス属性を直接参照。
- 参照: `kaggle/train_notebook.py` 577-579行。
- 影響: 実行中に複数設定を比較する際に状態混線しやすい。

2. デフォルト実行導線は `worst_case_v5`。
- 参照: `kaggle/train_notebook.py` 3299-3302行。
- 影響: `train_kfold`改修だけでは本番実行に反映されない。

## (4) 見落とし手法の検討

結論: このラウンドは「高コスト新規手法」より「評価健全化 + 低コスト改善」を優先。

手法評価:

1. `scSE`（推奨）
- 現行SMP構成に低コストで投入可能（`decoder_attention_type='scse'`）。
- 実装差分が小さく、最初の追加候補に適する。

2. EMA（推奨）
- SWAに近い安定化効果。SWAが不安定なら代替候補。
- 学習中に指数移動平均モデルを保持し、終端評価で比較。

3. Mixup（条件付き）
- 画像変換では過剰混合で境界がぼけやすい。
- 試すなら `alpha<=0.2` の弱設定、`input/target/mask` を同係数で同時混合。

4. CutMix（非優先）
- 局所パッチ置換で形態連続性を壊しやすく、本タスク適性が低い。

5. CBAM（非優先）
- 現コードへの組込みコストが高く、バグリスクに対する即効性が低い。

6. SAM（条件付き）
- 汎化改善余地はあるが、1step追加で時間コストが高い。
- 6h制約では優先度は低め。

## このラウンドの実行優先順位

1. `train_kfold`/`worst_case_v5` の「best checkpoint基準」修正。
2. SWA導入（本命導線にも適用）。
3. Progressive `256→512`。
4. `scSE`。
5. EMAまたは弱MixupのAB 1-2本。
