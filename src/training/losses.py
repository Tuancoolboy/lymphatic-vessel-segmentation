"""
Loss functions: BCE, Dice, Boundary, and a Combined loss
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

__all__ = ['BCELoss', 'DiceLoss', 'BoundaryLoss', 'CombinedLoss', 'DeepSupervisionLoss']


class DeepSupervisionLoss(nn.Module):
    """
    Loss that applies a criterion to multiple outputs and combines them with weights.
    Handles both deep supervision (tuple of outputs) and single output.
    Resizes the target to match the prediction size for each auxiliary output.
    """
    def __init__(self, criterion: nn.Module, aux_weights: list[float] = [0.4, 0.4, 0.2, 0.2]):
        super().__init__()
        self.criterion = criterion
        self.aux_weights = aux_weights

    def forward(self, predictions: torch.Tensor | tuple[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Can be a single tensor (for validation) or a tuple of tensors (for training).
                         The first tensor in the tuple is considered the main output.
            targets: The ground truth tensor.
        """
        # If it's a single tensor (validation case), just compute the loss
        if not isinstance(predictions, (list, tuple)):
            return self.criterion(predictions, targets)
            
        # The first prediction is the main one, the rest are auxiliary
        main_prediction = predictions[0]
        aux_predictions = predictions[1:]
        
        # Loss for the main output (weight is implicitly 1.0)
        loss = self.criterion(main_prediction, targets)
        
        # Add weighted losses for auxiliary outputs
        if len(aux_predictions) > 0:
            if len(self.aux_weights) != len(aux_predictions):
                print(f"Warning: Number of aux_weights ({len(self.aux_weights)}) does not match number of auxiliary predictions ({len(aux_predictions)}).")
            
            for i, p in enumerate(aux_predictions):
                if i < len(self.aux_weights) and self.aux_weights[i] > 0:
                    target_resized = F.interpolate(targets, size=p.shape[2:], mode='bilinear', align_corners=False)
                    loss += self.aux_weights[i] * self.criterion(p, target_resized)
        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(predictions, targets)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(predictions)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_level_set_function(gt_mask: torch.Tensor) -> torch.Tensor:
        gt_np = gt_mask.detach().cpu().numpy()
        phi_maps = []
        for i in range(gt_np.shape[0]):
            mask = gt_np[i, 0]
            if not mask.any():
                phi_maps.append(np.zeros_like(mask))
                continue
            dist_inside = distance_transform_edt(mask)
            dist_outside = distance_transform_edt(1 - mask)
            phi = dist_outside - dist_inside
            denom = np.max(np.abs(phi)) + 1e-8
            phi = phi / denom
            phi_maps.append(phi)
        phi = torch.tensor(np.stack(phi_maps), dtype=torch.float32, device=gt_mask.device)
        return phi.unsqueeze(1)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(predictions)
        phi_G = self.compute_level_set_function(targets)
        return torch.mean(phi_G * probs)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        alpha_schedule: str = 'rebalance',
        initial_alpha: float = 0.005,
        alpha_increment: float = 0.005
    ):
        super().__init__()
        self.boundary_loss = BoundaryLoss()
        self.dice_loss = DiceLoss()
        self.bce_loss = BCELoss()
        self.alpha_schedule = alpha_schedule
        self.initial_alpha = initial_alpha
        self.alpha_increment = alpha_increment
        self.current_epoch = 0
        self.alpha = initial_alpha

    def update_alpha(self, epoch: int):
        self.current_epoch = epoch
        if self.alpha_schedule == 'constant':
            self.alpha = self.initial_alpha
        elif self.alpha_schedule == 'increase':
            self.alpha = min(self.initial_alpha + self.alpha_increment * epoch, 1.0)
        elif self.alpha_schedule == 'rebalance':
            self.alpha = min(self.initial_alpha + self.alpha_increment * epoch, 1.0)
        else:
            self.alpha = self.initial_alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        
        # Optimization: Only compute boundary loss (expensive CPU op) if it contributes to the loss
        boundary = torch.tensor(0.0, device=predictions.device)
        
        if self.alpha_schedule == 'rebalance':
            # (1-β)*Dice + β*Boundary + 0.4*BCE ; β = (0.6 - alpha)
            beta = max(0.0, 0.6 - self.alpha)
            if beta > 0:
                boundary = self.boundary_loss(predictions, targets)
            combined = (1 - beta) * dice + beta * boundary + 0.4 * bce
        else:
            if self.alpha > 0:
                boundary = self.boundary_loss(predictions, targets)
            combined = dice + self.alpha * boundary + bce
        return combined