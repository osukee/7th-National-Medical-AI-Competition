"""
Transfer Learning Module: VirtualStaining → Medical AI Competition
Encoder再利用 + Decoder再構築 + 差分学習率

Strategy:
- Encoder: Load from VirtualStaining, fine-tune with low LR (1e-5)
- Bottleneck: Load, fine-tune with medium LR (5e-5)
- Decoder: Fresh initialization, full LR (1e-4)
- Output: Fresh initialization, full LR (1e-4)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_virtualstaining_encoder(
    pretrained_path: Path,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    VirtualStaining学習済みモデルからEncoder重みを抽出.
    
    Args:
        pretrained_path: VirtualStaining best_model.pth のパス
        device: デバイス
    
    Returns:
        Encoder層の state_dict
    """
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Encoder関連の重みのみ抽出
    encoder_keys = [k for k in checkpoint.keys() if 'encoder' in k.lower() or 'enc' in k.lower()]
    encoder_state = {k: v for k, v in checkpoint.items() if k in encoder_keys}
    
    print(f"Loaded {len(encoder_state)} encoder layers from VirtualStaining model")
    return encoder_state


def create_transfer_model(
    encoder_weights: Optional[Dict[str, torch.Tensor]] = None,
    in_channels: int = 1,
    out_channels: int = 1,
    freeze_encoder: bool = False
) -> nn.Module:
    """
    転移学習用モデルを作成.
    
    - Encoderは VirtualStaining の重みでウォームスタート
    - Decoderは新規初期化
    
    Args:
        encoder_weights: VirtualStaining encoder重み
        freeze_encoder: Encoderを完全凍結するか (非推奨)
    """
    try:
        import segmentation_models_pytorch as smp
        
        # SMPのU-Netを使用
        model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",  # まずImageNetで初期化
            in_channels=in_channels,
            classes=out_channels,
        )
        
        # VirtualStaining encoder重みで上書き
        if encoder_weights:
            current_state = model.encoder.state_dict()
            loaded_count = 0
            
            for key, value in encoder_weights.items():
                # キー名の変換（異なるモデル間でのマッピング）
                clean_key = key.replace("encoder.", "")
                if clean_key in current_state and current_state[clean_key].shape == value.shape:
                    current_state[clean_key] = value
                    loaded_count += 1
            
            model.encoder.load_state_dict(current_state)
            print(f"Loaded {loaded_count} layers from VirtualStaining encoder")
        
        # Encoderの凍結（非推奨だが選択可能）
        if freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("⚠️ Encoder frozen (not recommended)")
        
        return model
    
    except ImportError:
        print("Warning: SMP not available, using SimpleUNet")
        return _create_simple_transfer_unet(encoder_weights, freeze_encoder)


def _create_simple_transfer_unet(
    encoder_weights: Optional[Dict[str, torch.Tensor]] = None,
    freeze_encoder: bool = False
) -> nn.Module:
    """SimpleUNet用の転移学習."""
    
    class TransferUNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Encoder (転移対象)
            self.enc1 = self._double_conv(1, 64)
            self.enc2 = self._double_conv(64, 128)
            self.enc3 = self._double_conv(128, 256)
            self.enc4 = self._double_conv(256, 512)
            self.pool = nn.MaxPool2d(2)
            
            # Bottleneck (転移対象, 微調整)
            self.bottleneck = self._double_conv(512, 1024)
            
            # Decoder (新規初期化)
            self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = self._double_conv(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = self._double_conv(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = self._double_conv(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = self._double_conv(128, 64)
            
            # Output (新規初期化)
            self.out_conv = nn.Conv2d(64, 1, 1)
        
        def _double_conv(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            
            d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
            
            return self.out_conv(d1)
        
        def get_encoder_params(self):
            """Encoder層のパラメータを返す."""
            return list(self.enc1.parameters()) + list(self.enc2.parameters()) + \
                   list(self.enc3.parameters()) + list(self.enc4.parameters())
        
        def get_bottleneck_params(self):
            """Bottleneck層のパラメータを返す."""
            return list(self.bottleneck.parameters())
        
        def get_decoder_params(self):
            """Decoder層 + 出力層のパラメータを返す."""
            params = []
            for module in [self.up4, self.dec4, self.up3, self.dec3, 
                          self.up2, self.dec2, self.up1, self.dec1, self.out_conv]:
                params.extend(list(module.parameters()))
            return params
    
    model = TransferUNet()
    
    # 重みをロード
    if encoder_weights:
        current_state = model.state_dict()
        for key, value in encoder_weights.items():
            if key in current_state and current_state[key].shape == value.shape:
                current_state[key] = value
        model.load_state_dict(current_state)
    
    if freeze_encoder:
        for param in model.get_encoder_params():
            param.requires_grad = False
    
    return model


def create_differential_optimizer(
    model: nn.Module,
    encoder_lr: float = 1e-5,
    bottleneck_lr: float = 5e-5,
    decoder_lr: float = 1e-4,
    weight_decay: float = 1e-5
) -> optim.Optimizer:
    """
    差分学習率のオプティマイザを作成.
    
    Args:
        model: 転移学習モデル
        encoder_lr: Encoder用低学習率 (1e-5推奨)
        bottleneck_lr: Bottleneck用中間学習率 (5e-5推奨)
        decoder_lr: Decoder + Output用高学習率 (1e-4推奨)
    """
    try:
        import segmentation_models_pytorch as smp
        
        # SMPモデルの場合
        param_groups = [
            {"params": model.encoder.parameters(), "lr": encoder_lr, "name": "encoder"},
            {"params": model.decoder.parameters(), "lr": decoder_lr, "name": "decoder"},
            {"params": model.segmentation_head.parameters(), "lr": decoder_lr, "name": "head"},
        ]
    except:
        # SimpleUNetの場合
        if hasattr(model, 'get_encoder_params'):
            param_groups = [
                {"params": model.get_encoder_params(), "lr": encoder_lr, "name": "encoder"},
                {"params": model.get_bottleneck_params(), "lr": bottleneck_lr, "name": "bottleneck"},
                {"params": model.get_decoder_params(), "lr": decoder_lr, "name": "decoder"},
            ]
        else:
            # フォールバック: 全パラメータ同一LR
            param_groups = [{"params": model.parameters(), "lr": decoder_lr}]
    
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # 学習率情報を表示
    print("Differential Learning Rates:")
    for group in param_groups:
        name = group.get("name", "unknown")
        lr = group["lr"]
        num_params = sum(p.numel() for p in group["params"] if p.requires_grad)
        print(f"  - {name}: LR={lr:.1e}, params={num_params:,}")
    
    return optimizer


def reset_batchnorm_statistics(model: nn.Module) -> None:
    """
    BatchNormの統計をリセット.
    
    VirtualStainingとコンペデータで分布が違うため、
    running_mean/running_varをリセットしてから再計算させる.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()
            module.momentum = 0.1  # デフォルトに戻す
    
    print("BatchNorm statistics reset")


# ==============================================================================
# 使用例
# ==============================================================================

def example_transfer_training():
    """転移学習の使用例."""
    
    # 1. VirtualStaining encoder重みをロード
    vs_weights = load_virtualstaining_encoder(
        Path("notebooks/best_model.pth")
    )
    
    # 2. 転移学習モデルを作成
    model = create_transfer_model(
        encoder_weights=vs_weights,
        freeze_encoder=False  # 凍結はNG、低LRで微調整が最適
    )
    
    # 3. BatchNorm統計をリセット
    reset_batchnorm_statistics(model)
    
    # 4. 差分学習率オプティマイザを作成
    optimizer = create_differential_optimizer(
        model,
        encoder_lr=1e-5,      # Encoder: 低速学習
        bottleneck_lr=5e-5,    # Bottleneck: 中速
        decoder_lr=1e-4,       # Decoder: 通常速度
    )
    
    # 5. 本番データで学習
    # train_competition_model(model, optimizer, train_loader, ...)
    
    return model, optimizer


if __name__ == "__main__":
    print("=== Transfer Learning Module ===")
    print("\nStrategy:")
    print("  Encoder   → Reuse from VirtualStaining (LR: 1e-5)")
    print("  Bottleneck→ Reuse, fine-tune (LR: 5e-5)")
    print("  Decoder   → Fresh init (LR: 1e-4)")
    print("  Output    → Fresh init (LR: 1e-4)")
    print("\n⚠️ Do NOT freeze encoder - low LR fine-tuning is better!")
    print("⚠️ Do NOT reuse output layer - intensity distribution differs!")
