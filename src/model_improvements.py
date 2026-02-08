"""
Model Improvement Module for VirtualStaining
Implements advanced training strategies to improve model accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List
import numpy as np

try:
    import segmentation_models_pytorch as smp
    USE_SMP = True
except ImportError:
    USE_SMP = False


# ==============================================================================
# Advanced Loss Functions
# ==============================================================================

class GradientLoss(nn.Module):
    """Gradient-based loss for edge preservation."""
    
    def __init__(self):
        super().__init__()
        # Sobel kernels
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        if self.sobel_x.device != pred.device:
            self.sobel_x = self.sobel_x.to(pred.device)
            self.sobel_y = self.sobel_y.to(pred.device)
        
        pred_grad_x = nn.functional.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = nn.functional.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = nn.functional.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = nn.functional.conv2d(target, self.sobel_y, padding=1)
        
        loss = nn.functional.l1_loss(pred_grad_x, target_grad_x) + nn.functional.l1_loss(pred_grad_y, target_grad_y)
        return loss


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better texture preservation."""
    
    def __init__(self, layers: List[int] = [3, 8, 15]):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features
        
        self.layers = layers
        self.slices = nn.ModuleList()
        
        prev = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:layer+1]))
            prev = layer + 1
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        # Convert grayscale to 3-channel
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        loss = 0
        x_pred = pred
        x_target = target
        
        for slice_module in self.slices:
            x_pred = slice_module(x_pred)
            x_target = slice_module(x_target)
            loss += nn.functional.mse_loss(x_pred, x_target)
        
        return loss


class EnhancedLoss(nn.Module):
    """Combined loss with MSE, SSIM, Gradient, and optional Perceptual loss."""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 1.0,
        grad_weight: float = 0.5,
        perceptual_weight: float = 0.0
    ):
        super().__init__()
        from train_virtualstaining import SSIMLoss
        
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.grad_loss = GradientLoss()
        
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.perceptual_weight = perceptual_weight
        
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
    
    def forward(self, pred, target):
        loss = 0
        loss += self.mse_weight * self.mse_loss(pred, target)
        loss += self.ssim_weight * self.ssim_loss(pred, target)
        loss += self.grad_weight * self.grad_loss(pred, target)
        
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            loss += self.perceptual_weight * self.perceptual_loss(pred, target)
        
        return loss


# ==============================================================================
# Advanced Architectures
# ==============================================================================

def create_advanced_model(
    encoder_name: str = "efficientnet-b4",
    architecture: str = "unetplusplus",
    in_channels: int = 1,
    out_channels: int = 1
) -> nn.Module:
    """Create advanced segmentation model."""
    
    if not USE_SMP:
        from train_virtualstaining import SimpleUNet
        print("Warning: SMP not available, using SimpleUNet")
        return SimpleUNet(in_channels, out_channels)
    
    arch_map = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "pan": smp.PAN,
    }
    
    arch_class = arch_map.get(architecture.lower(), smp.Unet)
    
    model = arch_class(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_channels,
    )
    
    return model


# ==============================================================================
# Data Augmentation
# ==============================================================================

class VirtualStainingAugmentation:
    """Augmentation pipeline for virtual staining."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        try:
            import albumentations as A
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=1),
                    A.GaussianBlur(blur_limit=(3, 5), p=1),
                ], p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            ], additional_targets={"target": "image"})
            self.has_albumentation = True
        except ImportError:
            self.has_albumentation = False
            print("Warning: albumentations not installed. Augmentation disabled.")
    
    def __call__(self, image: np.ndarray, target: np.ndarray):
        if not self.has_albumentation or np.random.random() > self.p:
            return image, target
        
        # Ensure proper shape for albumentations
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if len(target.shape) == 2:
            target = target[:, :, np.newaxis]
        
        transformed = self.transform(image=image, target=target)
        aug_image = transformed["image"]
        aug_target = transformed["target"]
        
        # Remove extra dimension if added
        if aug_image.shape[-1] == 1:
            aug_image = aug_image[:, :, 0]
        if aug_target.shape[-1] == 1:
            aug_target = aug_target[:, :, 0]
        
        return aug_image, aug_target


# ==============================================================================
# Training Utilities
# ==============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """Cosine schedule with linear warmup."""
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# Recommended Training Configuration
# ==============================================================================

IMPROVEMENT_CONFIG = {
    "baseline": {
        "encoder": "resnet34",
        "architecture": "unet",
        "mse_weight": 1.0,
        "ssim_weight": 1.0,
        "grad_weight": 0.0,
        "epochs": 50,
        "lr": 1e-4,
    },
    "improved_v1": {
        "encoder": "efficientnet-b4",
        "architecture": "unet",
        "mse_weight": 1.0,
        "ssim_weight": 1.0,
        "grad_weight": 0.5,
        "epochs": 100,
        "lr": 1e-4,
    },
    "improved_v2": {
        "encoder": "efficientnet-b4",
        "architecture": "unetplusplus",
        "mse_weight": 1.0,
        "ssim_weight": 1.0,
        "grad_weight": 0.5,
        "epochs": 100,
        "lr": 5e-5,
    },
    "best": {
        "encoder": "efficientnet-b5",
        "architecture": "unetplusplus",
        "mse_weight": 1.0,
        "ssim_weight": 1.5,
        "grad_weight": 0.5,
        "epochs": 150,
        "lr": 3e-5,
        "use_augmentation": True,
        "early_stopping_patience": 20,
    }
}


if __name__ == "__main__":
    print("Available improvement configurations:")
    for name, config in IMPROVEMENT_CONFIG.items():
        print(f"\n{name}:")
        for k, v in config.items():
            print(f"  {k}: {v}")
