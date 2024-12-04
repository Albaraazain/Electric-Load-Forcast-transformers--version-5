"""
Trainer class for the Informer model.
Handles training loop, validation, and model checkpointing.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm
import numpy as np

from model.informer import Informer
from training.scheduler import create_scheduler

class Trainer:
    def __init__(
            self,
            model: Informer,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            config: dict,
            device: torch.device
    ):
        """
        Initialize trainer

        Args:
            model: Informer model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            criterion: Loss function
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device

        # Initialize scheduler
        self.scheduler = create_scheduler(
            optimizer=optimizer,
            num_epochs=config['max_epochs'],
            steps_per_epoch=len(train_loader),
            warmup_epochs=config.get('warmup_epochs', 3)
        )

        # Initialize mixed precision training
        self.scaler = GradScaler()
        
        logging.basicConfig(
            level=logging.DEBUG,  # Change to DEBUG level
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('debug.log'),
                logging.StreamHandler()
            ]
        )


        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_model_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"Saved best model checkpoint to {best_model_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train_epoch(self) -> float:
        """Train one epoch with improved logging"""
        self.model.train()
        total_loss = 0
        running_loss = 0.0  # Add this line to initialize running_loss
        log_interval = 50

        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, (encoder_inputs, decoder_inputs, targets) in enumerate(pbar):
                # Debug information
                if batch_idx == 0:
                    self.logger.info(f"\nBatch shapes:")
                    self.logger.info(f"Encoder inputs: {encoder_inputs.shape}")
                    self.logger.info(f"Decoder inputs: {decoder_inputs.shape}")
                    self.logger.info(f"Targets: {targets.shape}")
                
                # Move data to device
                encoder_inputs = encoder_inputs.to(self.device)
                decoder_inputs = decoder_inputs.to(self.device)
                targets = targets.to(self.device)

                # Clear gradients
                self.optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast():
                    outputs, _ = self.model(encoder_inputs)
                    if batch_idx == 0:
                        self.logger.info(f"Model output shape: {outputs.shape}")
                        self.logger.info(f"Target shape for loss: {targets.shape}")
                        
                    loss = self.criterion(outputs, targets)

                # Mixed precision backward pass
                self.scaler.scale(loss).backward() # type: ignore
                
                # Gradient clipping
                if self.config.get('grad_clip'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_( # type: ignore
                        self.model.parameters(),
                        self.config['grad_clip_value']
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Scheduler step
                self.scheduler.step()

                # Update metrics
                running_loss += loss.item()
                total_loss += loss.item()
                self.global_step += 1

                # Update progress bar with minimal info
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })

                # Periodic detailed logging
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f"\n{'='*30} Batch {batch_idx + 1}/{len(self.train_loader)} {'='*30}\n"
                        f"Average Loss: {avg_loss:.4f} | Learning Rate: {lr:.6f}\n"
                        f"{'='*80}"
                    )
                    running_loss = 0.0

        return total_loss / len(self.train_loader)

    def train(self):
        """Main training loop with improved logging"""
        self.logger.info("\n" + "="*80)
        self.logger.info("Starting Training")
        self.logger.info("="*80)

        for epoch in range(self.current_epoch, self.config['max_epochs']):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Epoch summary
            self.logger.info(
                f"\n{'='*30} Epoch {epoch} Summary {'='*30}\n"
                f"Train Loss: {train_loss:.4f}\n"
                f"Val Loss: {val_loss:.4f}\n"
                f"Best Val Loss: {self.best_val_loss:.4f}\n"
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}\n"
                f"{'='*80}"
            )

            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.logger.info(f"New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(is_best)

            # Early stopping
            if self.patience_counter >= self.config['patience']:
                self.logger.info(
                    f"\n{'='*30} Early Stopping {'='*30}\n"
                    f"No improvement for {self.config['patience']} epochs\n"
                    f"Best Val Loss: {self.best_val_loss:.4f}\n"
                    f"{'='*80}"
                )
                break

        self.logger.info("\n" + "="*80)
        self.logger.info("Training Completed!")
        self.logger.info("="*80)
        return self.train_losses, self.val_losses


    @torch.no_grad()
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0

        for encoder_inputs, decoder_inputs, targets in self.val_loader:
            # Move data to device
            encoder_inputs = encoder_inputs.to(self.device)
            decoder_inputs = decoder_inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs, _ = self.model(encoder_inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

