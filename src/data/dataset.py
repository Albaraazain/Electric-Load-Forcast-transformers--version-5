"""
Dataset implementation for the Informer model.

Dependencies:
- torch>=2.4.1
- numpy>=1.24.3
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import numpy as np

__version__ = '1.0.0'

class InformerDataset(Dataset):
    """
    Dataset for Informer model with sliding window
    """
    def __init__(
            self,
            data: torch.Tensor,
            input_window: int,
            prediction_window: int,
            stride: int = 1
    ):
        super().__init__()
        self.data = data
        self.input_window = input_window
        self.prediction_window = prediction_window
        self.stride = stride
        
        print(f"Dataset initialized with:")
        print(f"Input window: {input_window}")
        print(f"Prediction window: {prediction_window}")
        print(f"Data shape: {data.shape}")
        
        # Calculate valid indices
        self.indices = self._generate_indices()
        
        # Validate tensor device and dtype
        self._validate_tensor()

    def _validate_tensor(self):
        """Validate input tensor properties"""
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("Data must be a PyTorch tensor")
        if self.data.dtype != torch.float32:
            self.data = self.data.float()

    def _generate_indices(self) -> list:
        """Generate valid start indices for windows"""
        valid_indices = []
        total_window = self.input_window + self.prediction_window

        for i in range(0, len(self.data) - total_window + 1, self.stride):
            valid_indices.append(i)

        return valid_indices

    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return len(self.indices)

    def __getitem__(self, idx: int):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.input_window + self.prediction_window
        
        # Get full sequence
        sequence = self.data[start_idx:end_idx]
        
        # Debug prints for first batch only
        if idx == 0:
            print(f"\nDataset Debug for batch 0:")
            print(f"Sequence shape: {sequence.shape}")
            
            # Split into input and target windows
            input_seq = sequence[:self.input_window]
            target_seq = sequence[self.input_window:]
            
            print(f"Input sequence shape: {input_seq.shape}")
            print(f"Target sequence full shape: {target_seq.shape}")
            
            # Calculate and show shapes before final adjustments
            decoder_input = target_seq[:-1]
            target = target_seq[1:, 0:1]
            
            print(f"Final decoder input shape: {decoder_input.shape}")
            print(f"Final target shape: {target.shape}")
            print("=" * 50)
        else:
            input_seq = sequence[:self.input_window]
            target_seq = sequence[self.input_window:]
            decoder_input = target_seq[:-1]
            target = target_seq[1:, 0:1]
        
        return input_seq, decoder_input, target
    
class DatasetSplitter:
    """Handles splitting data into train/val/test sets"""

    @staticmethod
    def split_data(
            data: torch.Tensor,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            test_ratio: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data into train/val/test sets

        Args:
            data: Input tensor to split
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing

        Returns:
            Tuple of (train_data, val_data, test_data)

        Raises:
            AssertionError: If ratios don't sum to 1
            TypeError: If input is not a PyTorch tensor
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        return train_data, val_data, test_data