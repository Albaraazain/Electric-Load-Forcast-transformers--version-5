"""
Learning rate scheduler for the Informer model.
Implements warmup and cosine decay scheduling.
"""
import warnings

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup and cosine decay
    """
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int,
            total_steps: int,
            min_lr: float = 1e-6,
            last_epoch: int = -1
    ):
        """
        Initialize scheduler

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list:
        """Calculate learning rate based on step"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        step = self.last_epoch

        # Warmup phase
        if step < self.warmup_steps:
            lr_factor = float(step) / float(max(1, self.warmup_steps))
            return [base_lr * lr_factor for base_lr in self.base_lrs]

        # Cosine decay phase
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))

        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

def create_scheduler(
        optimizer: Optimizer,
        num_epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 3,
        min_lr: float = 1e-6
) -> WarmupCosineScheduler:
    """
    Create a learning rate scheduler

    Args:
        optimizer: Optimizer to schedule
        num_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate

    Returns:
        Configured learning rate scheduler
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    return WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr
    )