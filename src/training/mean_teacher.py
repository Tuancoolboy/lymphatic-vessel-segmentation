"""
This module implements the Mean Teacher algorithm for semi-supervised learning.
The student model is trained via gradient descent, while the teacher model's weights
are an Exponential Moving Average (EMA) of the student's weights. A consistency
loss is enforced between the student's and teacher's predictions on unlabeled data.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..utils.logging import TrainingLogger
from src.config import ModelConfig, TrainingConfig
from src.training.trainer import Trainer
        
from itertools import cycle

class MeanTeacherTrainer(Trainer):
    """
    Extends the base Trainer to implement the Mean Teacher algorithm for
    semi-supervised learning.

    Attributes:
        student_model (nn.Module): The primary model being trained with gradient descent.
        teacher_model (nn.Module): The model whose weights are an EMA of the student's.
                                   It provides more stable pseudo-labels.
        optimizer (torch.optim.Optimizer): The optimizer for the student model.
        consistency_loss (nn.Module): The loss function (e.g., MSE) used to measure
                                      the difference between student and teacher predictions.
        consistency_rampup (int): The number of epochs over which the consistency loss
                                  weight is linearly increased to its full value.
        consistency_weight (float): The maximum weight of the consistency loss.
        ema_decay (float): The decay rate for the teacher model's EMA update.
    """
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        model_config: ModelConfig,
        config: TrainingConfig,
        logger: TrainingLogger,
        device: str = "cuda"
    ):
        """
        Initializes the MeanTeacherTrainer.

        Args:
            student_model (nn.Module): The student model instance.
            teacher_model (nn.Module): The teacher model instance.
            optimizer (torch.optim.Optimizer): Optimizer for the student model.
            model_config (ModelConfig): Configuration for the model architecture.
            config (TrainingConfig): Configuration for the training process.
            logger (TrainingLogger): Logger for recording metrics.
            device (str): The computing device.
        """
        # Initialize the base Trainer with the student model.
        # The base class will handle device placement and criterion building.
        super().__init__(student_model, model_config, config, logger, device)
        
        self.student_model = self.model  # Rename for clarity
        self.teacher_model = teacher_model.to(self.device)
        self.optimizer = optimizer

        # The teacher model should not be updated by gradients.
        for param in self.teacher_model.parameters():
            param.detach_()

        # Loss for enforcing consistency between student and teacher predictions.
        self.consistency_loss = nn.MSELoss()

        # Mean Teacher specific hyperparameters
        self.consistency_rampup = config.consistency_rampup
        self.consistency_weight = config.consistency_weight
        self.ema_decay = config.ema_decay

    def update_teacher(self):
        """
        Update the teacher model's weights using an Exponential Moving Average (EMA)
        of the student model's weights.

        Formula: Teacher = ema_decay * Teacher + (1 - ema_decay) * Student
        """
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
                teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1 - self.ema_decay)
    
    def get_consistency_weight(self, epoch: int) -> float:
        """
        Calculates the consistency loss weight with a linear ramp-up.
        The weight increases from 0 to `self.consistency_weight` over `self.consistency_rampup` epochs.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The calculated consistency weight for the current epoch.
        """
        if epoch < self.consistency_rampup:
            return self.consistency_weight * (epoch / self.consistency_rampup)
        return self.consistency_weight

    def train_epoch(self, train_loaders: Tuple[DataLoader, DataLoader], optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        """
        Performs a single training epoch for the Mean Teacher model.

        This involves processing both labeled data for a supervised loss and
        unlabeled data for a consistency loss.

        Args:
            train_loaders (Tuple[DataLoader, DataLoader]): A tuple containing the
                                                           (labeled_loader, unlabeled_loader).
            optimizer (torch.optim.Optimizer): The optimizer for the student model.
            epoch (int): The current epoch number.

        Returns:
            Dict[str, float]: A dictionary of training metrics for the epoch.
        """
        labeled_loader, unlabeled_loader = train_loaders

        self.student_model.train()
        self.teacher_model.eval()  # Teacher is always in evaluation mode.
        
        total_loss, total_supervised_loss, total_consistency_loss = 0.0, 0.0, 0.0
        
        consistency_weight = self.get_consistency_weight(epoch)
        
        # Ensure we can iterate through the larger dataset completely.
        # The smaller dataset will be re-shuffled and iterated as needed.
        if len(labeled_loader) > len(unlabeled_loader):
            num_batches = len(labeled_loader)
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = cycle(iter(unlabeled_loader))
        else:
            num_batches = len(unlabeled_loader)
            unlabeled_iter = iter(unlabeled_loader)
            labeled_iter = cycle(iter(labeled_loader))
        
        with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.config.epochs} [TRAIN]") as pbar:
            for batch_idx in pbar:
                supervised_loss, consistency_loss = None, None

                # 1. Supervised Loss (from labeled data)
                try:
                    labeled_batch = next(labeled_iter)
                    images = labeled_batch['image'].to(self.device)
                    masks = labeled_batch['mask'].to(self.device)
                    student_pred = self.student_model(images)
                    supervised_loss = self.criterion(student_pred, masks.float())
                except StopIteration:
                    pass # Handled by the iterator recreation logic

                # 2. Consistency Loss (from unlabeled data)
                if consistency_weight > 0:
                    try:
                        unlabeled_batch = next(unlabeled_iter)
                        images_unlabeled = unlabeled_batch['image'].to(self.device)
                        
                        # Get student's prediction on unlabeled data
                        student_pred_unlabeled = self.student_model(images_unlabeled)
                        student_prob = torch.sigmoid(student_pred_unlabeled[0] if isinstance(student_pred_unlabeled, (list, tuple)) else student_pred_unlabeled)
                        
                        # Get teacher's prediction (pseudo-label) with no gradient
                        with torch.no_grad():
                            teacher_pred_unlabeled = self.teacher_model(images_unlabeled)
                            teacher_prob = torch.sigmoid(teacher_pred_unlabeled[0] if isinstance(teacher_pred_unlabeled, (list, tuple)) else teacher_pred_unlabeled)
                        
                        consistency_loss = self.consistency_loss(student_prob, teacher_prob)
                    except StopIteration:
                        pass # Handled by the iterator recreation logic

                # 3. Combine losses and update student model
                total_loss_batch = torch.tensor(0.0, device=self.device)
                if supervised_loss is not None:
                    total_loss_batch += supervised_loss
                if consistency_loss is not None:
                    total_loss_batch += consistency_weight * consistency_loss
                
                if total_loss_batch.item() > 0:
                    optimizer.zero_grad()
                    total_loss_batch.backward()
                    optimizer.step()

                # 4. Update teacher model weights via EMA
                self.update_teacher()
                
                # 5. Log batch metrics
                total_loss += total_loss_batch.item()
                if supervised_loss is not None:
                    total_supervised_loss += supervised_loss.item()
                if consistency_loss is not None:
                    total_consistency_loss += consistency_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{total_loss_batch.item():.4f}',
                    'sup': f'{supervised_loss.item():.4f}' if supervised_loss else '0',
                    'cons': f'{consistency_loss.item():.4f}' if consistency_loss else '0',
                    'λ': f'{consistency_weight:.2f}'
                })
        
        return {
            'loss': total_loss / num_batches,
            'supervised_loss': total_supervised_loss / num_batches,
            'consistency_loss': total_consistency_loss / num_batches,
            'consistency_weight': consistency_weight
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validates the model. For Mean Teacher, validation is performed on the
        teacher model, as it's expected to be more stable and perform better.

        Args:
            val_loader (DataLoader): DataLoader for the validation set.

        Returns:
            Dict[str, float]: A dictionary of validation metrics.
        """
        # Validate using the more stable teacher model
        return super().validate(val_loader, model_to_validate=self.teacher_model)
    
    def train(self, labeled_loader: DataLoader, unlabeled_loader: Optional[DataLoader], val_loader: DataLoader, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> nn.Module:
        """
        Orchestrates the main training loop for the Mean Teacher setup.

        This method adapts the base `Trainer.train` loop by passing both labeled and
        unlabeled data loaders to the `train_epoch` method.

        Args:
            labeled_loader (DataLoader): DataLoader for labeled data.
            unlabeled_loader (Optional[DataLoader]): DataLoader for unlabeled data.
            val_loader (DataLoader): DataLoader for validation data.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler.

        Returns:
            nn.Module: The best teacher model found during training.
        """
        train_loaders = (labeled_loader, unlabeled_loader)
        
        # The base `train` method handles the epoch loop, validation, logging, and early stopping.
        # Our overridden `train_epoch` and `validate` methods ensure the Mean Teacher
        # logic is correctly applied within that loop.
        super().train(train_loaders, val_loader, self.optimizer, scheduler)
        
        # Return the teacher model, as it's the final, more stable artifact.
        print("--- Training finished. Returning the best TEACHER model. ---")
        return self.teacher_model